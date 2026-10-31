#!/usr/bin/env python3
"""codeflux-rig — autonomous LLM diff agent (fork-review doctrine in a loop).

Pipeline per task:
  snapshot target files -> generate_diff (configurable model chain) ->
  smartapply rescue -> verify (AST/compile/tests) -> code-run-first satisfaction
  gate (agentic-completions) -> outbox proposal (dry-run) or apply.

Config resolution (obvious and layered):
  1. --config path (YAML)
  2. env RIG_CONFIG
  3. ./rig.yml next to the script
  4. built-in defaults

Config keys (all optional):
  model: glm-5.3-flash          # primary freeinference model
  fallback_models: [qwen3.6-35b, deepseek-v4-flash]
  max_attempts: 3               # diff attempts per task (retry with feedback)
  mode: dry-run                 # dry-run | apply | apply-push
  interval_minutes: 30          # serve-loop cadence
  outbox: <dir>                 # proposals land here (default: ./outbox)
  tasks:                        # static task list
    - repo: /path/to/repo
      files: [relative/paths.py]
      goal: "..."
  discovery: none               # none | dirty-repos (later: more probes)

Env:
  FREEINFERENCE_API_KEY  (or secretsmith keyring label of same name)
  GPTDIFF_LLM_BASE_URL   (default https://freeinference.org/v1)
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import pathlib
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile

import yaml

RIG_HOME = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(RIG_HOME.parent / "agentic-completions"))
sys.path.insert(0, str(RIG_HOME.parent / "codeflux-patchling"))

DEFAULTS = {
    "model": "glm-5.3-flash",
    "fallback_models": ["qwen3.6-35b", "deepseek-v4-flash"],
    "max_attempts": 3,
    "mode": "dry-run",
    "interval_minutes": 30,
    "outbox": str(RIG_HOME / "outbox"),
    "discovery": "none",
    "tasks": [],
}


def load_config(path: str | None) -> dict:
    candidates = [path, os.environ.get("RIG_CONFIG"), str(RIG_HOME / "rig.yml")]
    for c in candidates:
        if c and os.path.isfile(c):
            cfg = yaml.safe_load(open(c)) or {}
            merged = dict(DEFAULTS)
            merged.update(cfg)
            return merged
    return dict(DEFAULTS)


def get_key() -> str:
    key = os.environ.get("FREEINFERENCE_API_KEY")
    if key:
        return key
    r = subprocess.run(["secretsmith", "get", "--label", "FREEINFERENCE_API_KEY", "--show"],
                       capture_output=True, text=True, timeout=30)
    for line in r.stdout.splitlines():
        if line.startswith("secret: "):
            return line.removeprefix("secret: ").strip()
    raise RuntimeError("no FREEINFERENCE_API_KEY (env or keyring)")


def model_call(messages: list[dict], cfg: dict, key: str) -> tuple[str, str]:
    """Try primary then fallbacks. Returns (text, model_used)."""
    from completions import chat
    for model in [cfg["model"], *cfg["fallback_models"]]:
        try:
            os.environ["FREEINFERENCE_API_KEY"] = key
            text, _ = chat(messages, model=model)
            if text.strip():
                return text, model
        except Exception as e:
            print(f"rig: model {model} failed: {e}", file=sys.stderr)
    raise RuntimeError("all models failed")


def snapshot(files: list[str], repo: str) -> dict[str, str]:
    return {f: (pathlib.Path(repo) / f).read_text() for f in files}


def feedback_for(diff: str, files: dict[str, str]) -> str:
    """Why a diff was rejected; fed back to the model on retry."""
    import patchling.core as pc
    parsed = pc.parse_diff_per_file(diff)
    notes = []
    for path, patch in parsed:
        if path is None or path not in files:
            notes.append(f"header path {path!r} does not match any real file")
        signal = [l[1:].strip() for l in patch.splitlines()
                  if l.startswith(("-", " ")) and not l.startswith(("--- ", "+++ "))]
        signal = [s for s in signal if s and len(s) > 3]
        if signal and path in files:
            hits = sum(1 for s in signal if s in files[path])
            if hits == 0:
                notes.append(f"context lines for {path} appear nowhere in the real file "
                             f"(hallucinated?) — use the exact lines provided")
    return "; ".join(notes) or "diff did not apply"


def semantic_hints(goal: str, diff: str, files: dict[str, str]) -> list[str]:
    """bge-m3 lane: rank files by cosine(goal+diff-signal, file content).
    Returns notes telling the model which file the goal most likely targets —
    used when header paths and substring content-match both fail."""
    from completions import embed, cosine
    signal = " ".join(l[1:].strip() for l in diff.splitlines()
                      if l.startswith(("-", " ")) and not l.startswith(("--- ", "+++ ")))[:4000]
    query = f"{goal}\n{signal}"
    names = list(files)
    try:
        vecs = embed([query] + [files[f][:6000] for f in names])
    except Exception as e:
        print(f"rig: bge lane unavailable: {e}", file=sys.stderr)
        return []
    q = vecs[0]
    ranked = sorted(zip(names, vecs[1:]), key=lambda kv: -cosine(q, kv[1]))
    top, score = ranked[0]
    notes = [f"semantic match (bge-m3 cos={score:.2f}): the goal most likely "
             f"targets file {top!r} — header your diff 'a/{top} b/{top}'"]
    return notes if score > 0.3 else []


def attempt_task(task: dict, cfg: dict, key: str) -> dict:
    """One task: up to max_attempts diff->rescue->verify rounds."""
    import patchling.core as pc
    repo = pathlib.Path(task["repo"]).resolve()
    names = task["files"]
    original = snapshot(names, str(repo))
    goal = task["goal"]
    feedback = ""
    for attempt in range(1, cfg["max_attempts"] + 1):
        excerpts = "\n\n".join(
            f"FILE {f}:\n```python\n{src[:12000]}\n```" for f, src in original.items())
        messages = [
            {"role": "system", "content":
                'Output a full unified git diff into a "```diff" block. '
                "NEVER remove or replace existing functions unless the goal says so. "
                "Use ONLY helpers and names that exist in the file. "
                "Context lines must be copied EXACTLY from the real file."},
            {"role": "user", "content": excerpts + f"\n\nGoal: {goal}" +
                (f"\n\nPrevious attempt rejected: {feedback}\nFix it." if feedback else "")},
        ]
        text, used = model_call(messages, cfg, key)
        m = re.search(r"```diff\n(.*?)```", text, re.S)
        if not m:
            m = re.search(r"(^--- .*\n.*\n@@.*)", text, re.S | re.M)
        diff = m.group(1).strip() if m else ""
        if not diff:
            feedback = "no diff block in response"
            continue
        out = pc.smartapply(diff, dict(original))
        changed = {k: v for k, v in out.items()
                   if k in original and v != original[k]}
        if not changed:
            feedback = feedback_for(diff, original)
            if not feedback or "nowhere in the real file" in feedback:
                hints = semantic_hints(goal, diff, original)
                if hints:
                    feedback = (feedback + "; " if feedback else "") + "; ".join(hints)
            print(f"  attempt {attempt}: rejected — {feedback}")
            continue
        return {"ok": True, "diff": diff, "files": changed, "model": used,
                "attempts": attempt, "original": original}
    return {"ok": False, "error": "max attempts exhausted", "model": cfg["model"]}


def verify(changed: dict[str, str], cfg: dict) -> tuple[bool, list[tuple[str, bool, str]]]:
    """CODE RUNS FIRST: AST every changed file; py_compile .py; optional pytest."""
    results = []
    ok_all = True
    with tempfile.TemporaryDirectory() as td:
        for name, content in changed.items():
            ast.parse(content)
            results.append((f"ast:{name}", True, "parsed"))
            if name.endswith(".py"):
                p = pathlib.Path(td) / pathlib.Path(name).name
                p.write_text(content)
                try:
                    py_compile.compile(str(p), doraise=True)
                    results.append((f"py_compile:{name}", True, "compiled"))
                except py_compile.PyCompileError as e:
                    ok_all = False
                    results.append((f"py_compile:{name}", False, str(e)[:200]))
        tests = cfg.get("verify_cmd")
        if tests and ok_all:
            r = subprocess.run(tests, shell=True, capture_output=True, text=True,
                               cwd=cfg.get("verify_cwd"), timeout=600)
            ok_all = r.returncode == 0
            results.append(("verify_cmd", ok_all, (r.stdout + r.stderr)[-300:]))
    return ok_all, results


def satisfaction(task: dict, result: dict, cfg: dict, key: str, check_rows) -> dict:
    """check_rows: pre-computed (name, ok, evidence) tuples from verify()."""
    from completions import satisfy
    checks = [(n, (lambda o=o, e=e: (o, e))) for n, o, e in check_rows]
    return satisfy(checks, judgment_prompt=
        f"Goal: {task['goal']}\nChanged: {list(result['files'])}\n"
        "Review the applied change for goal-fit and regressions.")


def run_task(task: dict, cfg: dict, key: str) -> dict:
    print(f"== task: {task['repo']} {task['files']} — {task['goal'][:60]}…")
    result = attempt_task(task, cfg, key)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
    slug = pathlib.Path(task["repo"]).name
    outdir = pathlib.Path(cfg["outbox"]) / f"{stamp}-{slug}"
    if not result["ok"]:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "failure.json").write_text(json.dumps(result, indent=2))
        print(f"  FAILED -> {outdir}/failure.json")
        return result
    ok, checks = verify(result["files"], cfg)
    sres = satisfaction(task, result, cfg, key, checks) if ok else \
        {"satisfied": False, "note": "code checks failed; model not consulted"}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "proposal.diff").write_text(result["diff"])
    (outdir / "changed").mkdir(exist_ok=True)
    for name, content in result["files"].items():
        (outdir / "changed" / pathlib.Path(name).name).write_text(content)
    (outdir / "verdict.json").write_text(json.dumps(
        {"ok": result["ok"], "model": result["model"], "attempts": result["attempts"],
         "code_checks": [(n, o, e) for n, o, e in checks],
         "satisfied": sres.get("satisfied"), "advisory": sres.get("advisory"),
         "judgment": sres.get("judgment")}, indent=2))
    print(f"  code_ok={ok} satisfied={sres.get('satisfied')} -> {outdir}")
    if cfg["mode"] in ("apply", "apply-push") and ok and sres.get("satisfied"):
        for name, content in result["files"].items():
            (pathlib.Path(task["repo"]) / name).write_text(content)
        print(f"  APPLIED to {task['repo']} (commit manually or apply-push mode)")
    return result


def discover_tasks(cfg: dict) -> list[dict]:
    if cfg.get("discovery") != "dirty-repos":
        return []
    tasks = []
    for pen in ("forge", "ranch", "geode", "olympus"):
        base = f"/home/toxic/pens/{pen}/products"
        if not os.path.isdir(base):
            continue
        for prod in os.listdir(base):
            repo = os.path.join(base, prod)
            if not os.path.isdir(os.path.join(repo, ".git")):
                continue
            r = subprocess.run(["git", "-C", repo, "status", "--short"],
                               capture_output=True, text=True, timeout=30)
            dirty = [l for l in r.stdout.splitlines() if l.endswith(".py")]
            if dirty:
                files = [l.split()[-1] for l in dirty][:3]
                tasks.append({"repo": repo, "files": files,
                              "goal": "Review and improve the dirty Python files: fix obvious defects, keep behavior."})
    return tasks


def serve(cfg: dict, key: str, once: bool = False) -> None:
    while True:
        tasks = list(cfg["tasks"]) + discover_tasks(cfg)
        for t in tasks:
            try:
                run_task(t, cfg, key)
            except Exception as e:
                print(f"rig: task error: {e}", file=sys.stderr)
        if once:
            return
        import time
        time.sleep(cfg["interval_minutes"] * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="codeflux-rig autonomous diff agent")
    ap.add_argument("--config")
    ap.add_argument("--once", action="store_true", help="run all tasks once and exit")
    ap.add_argument("--serve", action="store_true", help="loop on interval")
    a = ap.parse_args()
    cfg = load_config(a.config)
    key = get_key()
    if a.serve or (not a.once):
        serve(cfg, key, once=a.once)
    else:
        serve(cfg, key, once=True)


if __name__ == "__main__":
    main()
