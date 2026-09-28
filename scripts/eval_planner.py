"""Measure how well a local model plans NEXUS turns.

Plans only - nothing is executed. The catalog is the real tools plus
launch_app/close_app *definitions* (v0.1 subsystem 4) so app/game phrasing can
be measured before those tools exist.

    .venv\\Scripts\\python.exe scripts\\eval_planner.py llama3.1:8b@gpu llama3.2@gpu
    .venv\\Scripts\\python.exe scripts\\eval_planner.py llama3.2@cpu --limit 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus.ai.conversation.engine import strip_wake_word  # noqa: E402
from nexus.ai.conversation.responder import claims_unverified  # noqa: E402
from nexus.ai.memory.sqlite_store import SQLiteMemoryStore  # noqa: E402
from nexus.ai.router import FastPathRouter  # noqa: E402
from nexus.ai.planner import LLMPlanner, PlanError  # noqa: E402
from nexus.ai.providers.ollama import OllamaChat  # noqa: E402
from nexus.core.resources import Placement  # noqa: E402
from nexus.system.apps.service import AppService  # noqa: E402
from nexus.system.platform import current  # noqa: E402
from nexus.tools import default_tools  # noqa: E402
from nexus.tools.base import Param, Risk, Tool, ToolResult  # noqa: E402


def _never(**_):
    raise AssertionError("evaluation must never execute tools")


EVAL_ONLY_TOOLS: list[Tool] = []  # every planned app tool (launch/close/aliases) exists now


def score(case: dict, plan) -> dict:
    tool = plan.actions[0].tool if plan and plan.actions else "none"
    tool_ok = tool in case["tools"]
    args = plan.actions[0].args if plan and plan.actions else {}
    text = " ".join(str(v) for v in args.values()).lower()
    wanted = case.get("contains", [])
    wanted = [wanted] if isinstance(wanted, str) else wanted
    contains_ok = tool == "none" or not wanted or any(w in text for w in wanted)
    section_ok = "section" not in case or args.get("section") in (case["section"], None, "all")
    return {
        "pass": tool_ok and contains_ok,
        "tool": tool, "args": args, "tool_ok": tool_ok, "args_ok": contains_ok,
        "section_exact": "section" not in case or args.get("section") == case["section"],
        "section_ok": section_ok,
        "lang_ok": None if "lang" not in case or plan is None else plan.language == case["lang"],
        # Only conversation-only replies are spoken; with actions NEXUS uses its own words.
        "claim": bool(plan and not plan.actions and claims_unverified(plan.say)),
        "say": plan.say if plan else None,
    }


def run(model_spec: str, cases: list[dict], catalog: str) -> dict:
    model, _, placement = model_spec.partition("@")
    llm = OllamaChat(model, placement=Placement(placement or "gpu"), ram_mb=0, idle_unload_s=120)
    if not llm.available():
        sys.exit(f"{model} is not installed in Ollama")
    apps = AppService(current())  # real discovery; tools are planned, never executed here
    tools = default_tools(current(), SQLiteMemoryStore(None), apps)  # in-RAM memory store: nothing saved
    if catalog == "planned":
        tools += EVAL_ONLY_TOOLS
    else:
        # Today's tools only: requests for missing tools must be declined, not substituted.
        names = {t.name for t in tools} | {"none"}
        cases = [c if set(c["tools"]) & names else {**c, "tools": ["none"], "contains": []} for c in cases]
    games = ", ".join(g.display_name for g in apps.catalog().games())
    planner = LLMPlanner(llm, tools, context=lambda: f"Installed games on this PC: {games}." if games else "")
    router = FastPathRouter(apps=apps)

    t0 = time.perf_counter()
    llm.load()
    load_s = time.perf_counter() - t0
    results = []
    try:
        for case in cases:
            started = time.perf_counter()
            try:
                plan, error = planner.plan(case["text"]), None
            except PlanError as exc:
                plan, error = None, str(exc)
            r = {"id": case["id"], "cat": case["cat"], "text": case["text"],
                 "seconds": round(time.perf_counter() - started, 2), "error": error, **score(case, plan)}
            # What NEXUS actually does: the no-model fast path when it takes the request, else the model.
            fast = router.route(strip_wake_word(case["text"]))
            r["fast_path"] = fast is not None
            r["system_pass"] = score(case, fast)["pass"] if fast is not None else r["pass"]
            results.append(r)
            mark = "ok " if r["pass"] else "BAD"
            print(f"  {mark} {r['seconds']:5.2f}s  {case['text'][:44]:<44} -> {r['tool']} {json.dumps(r['args'], ensure_ascii=False)[:60]}"
                  + ("   [CLAIM]" if r["claim"] else ""), flush=True)
    finally:
        llm.unload()

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["cat"]].append(r["pass"])
    lat = [r["seconds"] for r in results]
    langs = [r["lang_ok"] for r in results if r["lang_ok"] is not None]
    summary = {
        "model": model_spec, "catalog": catalog, "cases": len(results), "load_s": round(load_s, 1),
        "pass_rate": round(sum(r["pass"] for r in results) / len(results), 3),
        "system_pass_rate (fast path + model)": round(sum(r["system_pass"] for r in results) / len(results), 3),
        "fast_path_share": f"{sum(r['fast_path'] for r in results)}/{len(results)}",
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in by_cat.items()},
        "section_exact": sum(r["section_exact"] for r in results if r["cat"] == "sysinfo"),
        "language_ok": f"{sum(langs)}/{len(langs)}",
        "claims_in_say": sum(r["claim"] for r in results),
        "parse_errors": sum(r["error"] is not None for r in results),
        "latency_median_s": round(statistics.median(lat), 2),
        "latency_p90_s": round(sorted(lat)[int(len(lat) * 0.9) - 1], 2),
    }
    return {"summary": summary, "results": results}


def router_report(cases: list[dict]) -> dict:
    """How many requests the no-model fast path answers, and whether it is ever wrong."""
    router = FastPathRouter(apps=AppService(current()))
    taken, wrong = [], []
    for case in cases:
        plan = router.route(case["text"])
        if plan is None:
            continue
        taken.append(case["id"])
        tool = plan.actions[0].tool
        if tool not in case["tools"]:
            wrong.append(f"{case['text']!r} -> {tool} (expected {case['tools']})")
    return {"cases": len(cases), "fast_path": len(taken), "wrong": wrong}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="model@placement, e.g. llama3.1:8b@gpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--catalog", choices=("planned", "production"), default="planned",
                    help="planned: include launch_app/close_app definitions; production: today's tools only")
    args = ap.parse_args()
    cases = json.loads((ROOT / "evals" / "planner_cases.json").read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[::max(1, len(cases) // args.limit)][:args.limit]
    out_dir = ROOT / "var" / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    fast = router_report(cases)
    print(f"fast path (no model): {fast['fast_path']}/{fast['cases']} requests, wrong: {len(fast['wrong'])}")
    for w in fast["wrong"]:
        print(f"  WRONG {w}")
    for spec in args.models:
        print(f"\n=== {spec}, {args.catalog} catalog ({len(cases)} cases) ===")
        report = run(spec, cases, args.catalog)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"planner-{spec.replace(':', '_').replace('@', '-')}-{args.catalog}-{stamp}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
        print(f"details: {path}")


if __name__ == "__main__":
    main()
