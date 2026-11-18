"""
Run the text-to-SQL evaluation harness.

    uv run python -m eval.run --predictor oracle
    uv run python -m eval.run --predictor naive --out eval/results/naive.json
    EVAL_LLM_API_KEY=... uv run python -m eval.run --predictor llm \
        --model anthropic/claude-sonnet-4.5 --out eval/results/claude.json

The only predictor that needs network or a key is ``llm``; ``oracle`` and
``naive`` run offline and are what CI executes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from server.tasks.definitions import SCHEMA_INFO

from .dataset import GOLDEN_SET
from .metrics import ItemResult, evaluate_item, summarise, summary_to_dict
from .predictors import build_predictor

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "data" / "schema.sql"


def fresh_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()
    return conn


def run(predictor_kind: str, model: str | None = None, verbose: bool = True) -> list[ItemResult]:
    predictor = build_predictor(predictor_kind, model=model)
    conn = fresh_connection()
    results: list[ItemResult] = []

    for item in GOLDEN_SET:
        started = time.perf_counter()
        try:
            predicted_sql = predictor.predict(item, SCHEMA_INFO)
        except Exception as exc:  # a model failure is data, not a crash
            predicted_sql = ""
            result = ItemResult(
                item_id=item.id,
                difficulty=item.difficulty,
                question=item.question,
                predicted_sql="",
                reward=0.0,
                exact_match=False,
                result_match=False,
                value_match=False,
                error=f"predictor error: {exc}",
            )
        else:
            result = evaluate_item(
                conn,
                item_id=item.id,
                difficulty=item.difficulty,
                question=item.question,
                predicted_sql=predicted_sql,
                reference_sql=item.reference_sql,
            )
        result.latency_s = round(time.perf_counter() - started, 2)
        results.append(result)
        if verbose:
            mark = "PASS" if result.exact_match else ("PART" if result.reward > 0 else "FAIL")
            print(f"  {item.id:4} {mark} reward={result.reward:.1f} "
                  f"({result.latency_s}s)")

    return results


def replay(path: str) -> list[ItemResult]:
    """
    Re-score stored predictions without calling a model.

    Keeps published numbers reproducible and cheap to audit when the metrics
    themselves change — the predictions are the expensive part, not the scoring.
    """
    payload = json.loads(Path(path).read_text())
    by_id = {item.id: item for item in GOLDEN_SET}
    conn = fresh_connection()
    results: list[ItemResult] = []
    for record in payload["results"]:
        item = by_id[record["item_id"]]
        result = evaluate_item(
            conn,
            item_id=item.id,
            difficulty=item.difficulty,
            question=item.question,
            predicted_sql=record["predicted_sql"],
            reference_sql=item.reference_sql,
        )
        result.latency_s = record.get("latency_s")
        results.append(result)
    return results


def render_markdown(summary, results: list[ItemResult]) -> str:
    data = summary_to_dict(summary)
    lo, hi = data["exact_match_ci95"]
    lines = [
        f"### {data['predictor']}",
        "",
        f"- items: **{data['n']}**",
        f"- exact match: **{data['exact_match_rate']:.1%}** "
        f"(95% CI {lo:.0%}–{hi:.0%})",
        f"- result match (same column names, row order ignored): "
        f"**{data['result_match_rate']:.1%}**",
        f"- value match (column names ignored): **{data['value_match_rate']:.1%}**",
        f"- mean graded reward: **{data['mean_reward']:.2f}**",
        f"- SQL error rate: **{data['error_rate']:.1%}**",
        "",
        "| difficulty | n | exact match | mean reward |",
        "|---|---:|---:|---:|",
    ]
    for difficulty, stats in data["per_difficulty"].items():
        lines.append(
            f"| {difficulty} | {stats['n']} | "
            f"{stats['exact_match_rate']:.1%} | {stats['mean_reward']:.2f} |"
        )

    failures = [r for r in results if not r.exact_match]
    if failures:
        lines += ["", "<details><summary>Failures</summary>", ""]
        for r in failures:
            reason = r.error or f"reward {r.reward:.1f}"
            lines.append(f"- **{r.item_id}** ({r.difficulty}) — {reason}")
        lines += ["", "</details>"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Text-to-SQL evaluation harness")
    parser.add_argument("--predictor", default="oracle",
                        choices=["oracle", "naive", "llm"])
    parser.add_argument("--model", default=None,
                        help="model id for the llm predictor")
    parser.add_argument("--out", default=None,
                        help="write JSON results to this path")
    parser.add_argument("--markdown", default=None,
                        help="write a markdown summary to this path")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--replay", default=None, metavar="PATH",
                        help="re-score predictions from a previous results JSON")
    args = parser.parse_args(argv)

    if args.replay:
        results = replay(args.replay)
        name = json.loads(Path(args.replay).read_text())["summary"]["predictor"]
    else:
        results = run(args.predictor, args.model, verbose=not args.quiet)
        name = args.predictor
    summary = summarise(
        f"llm:{args.model}" if args.predictor == "llm" and args.model
        else name,
        results,
    )

    payload = {
        "summary": summary_to_dict(summary),
        "results": [
            {
                "item_id": r.item_id,
                "difficulty": r.difficulty,
                "question": r.question,
                "predicted_sql": r.predicted_sql,
                "reward": r.reward,
                "exact_match": r.exact_match,
                "result_match": r.result_match,
                "value_match": r.value_match,
                "error": r.error,
                "latency_s": r.latency_s,
            }
            for r in results
        ],
    }

    print()
    print(render_markdown(summary, results))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {out}")

    if args.markdown:
        md = Path(args.markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(summary, results) + "\n")
        print(f"wrote {md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
