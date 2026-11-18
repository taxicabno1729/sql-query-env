"""
Scoring layer for the text-to-SQL evaluation harness.

Two things are deliberately kept separate:

* the **graded reward** reuses the environment's own grader
  (``server.tasks.grader.grade``), so evaluation and training reward can never
  drift apart;
* the **hard metrics** (exact match, unordered result match, error rate) are
  computed here and are what the published numbers use, because a partial-credit
  reward is not an accuracy.

``exact_match`` requires the same columns and the same rows *in the same order*.
``result_match`` ignores row order but still requires identical columns and
multiset of rows. Reporting both makes order-sensitivity visible instead of
hiding it behind one number.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from server.tasks.grader import grade


@dataclass
class ItemResult:
    item_id: str
    difficulty: str
    question: str
    predicted_sql: str
    reward: float            # graded 0.0–1.0 from the environment grader
    exact_match: bool        # same column names + same rows, same order
    result_match: bool       # same column names + same rows, any order
    value_match: bool        # same row values, column names ignored
    error: str | None = None
    latency_s: float | None = None


@dataclass
class Summary:
    predictor: str
    n: int
    exact_match_rate: float
    result_match_rate: float
    value_match_rate: float
    mean_reward: float
    error_rate: float
    exact_match_ci: tuple[float, float]
    per_difficulty: dict[str, dict[str, Any]] = field(default_factory=dict)


def _normalise(rows: list[tuple]) -> list[tuple]:
    """Round floats to 2 dp so 2.0 and 2.0000000001 compare equal."""
    out = []
    for row in rows:
        out.append(
            tuple(round(v, 2) if isinstance(v, float) else v for v in row)
        )
    return out


def _run(conn: sqlite3.Connection, sql: str) -> tuple[list[str], list[tuple]]:
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    return cols, _normalise(cur.fetchall())


def evaluate_item(
    conn: sqlite3.Connection,
    item_id: str,
    difficulty: str,
    question: str,
    predicted_sql: str,
    reference_sql: str,
) -> ItemResult:
    """Score one prediction. Never raises: a bad query is a recorded failure."""
    graded = grade(conn, predicted_sql, reference_sql)

    exact = False
    result_match = False
    value_match = False

    try:
        got_cols, got_rows = _run(conn, predicted_sql)
        exp_cols, exp_rows = _run(conn, reference_sql)
        cols_equal = [c.lower() for c in got_cols] == [c.lower() for c in exp_cols]
        if cols_equal:
            exact = got_rows == exp_rows
            result_match = Counter(got_rows) == Counter(exp_rows)
        # Column names are a style choice; values are the answer. This metric
        # separates "wrong query" from "right query, different alias".
        value_match = len(got_cols) == len(exp_cols) and Counter(got_rows) == Counter(exp_rows)
    except sqlite3.Error:
        # graded.error already carries the message
        pass

    return ItemResult(
        item_id=item_id,
        difficulty=difficulty,
        question=question,
        predicted_sql=predicted_sql,
        reward=graded.score,
        exact_match=exact,
        result_match=result_match,
        value_match=value_match,
        error=graded.error,
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a proportion.

    Reported alongside the point estimate because the golden set is small: the
    interval is the honest way to say "12 items" without implying precision the
    sample cannot support.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


def summarise(predictor: str, results: list[ItemResult]) -> Summary:
    n = len(results)
    if n == 0:
        raise ValueError("no results to summarise")

    exact = sum(r.exact_match for r in results)
    result_match = sum(r.result_match for r in results)
    errors = sum(1 for r in results if r.error)

    per_difficulty: dict[str, dict[str, Any]] = {}
    for difficulty in sorted({r.difficulty for r in results}):
        subset = [r for r in results if r.difficulty == difficulty]
        per_difficulty[difficulty] = {
            "n": len(subset),
            "exact_match_rate": sum(r.exact_match for r in subset) / len(subset),
            "mean_reward": sum(r.reward for r in subset) / len(subset),
        }

    return Summary(
        predictor=predictor,
        n=n,
        exact_match_rate=exact / n,
        result_match_rate=result_match / n,
        value_match_rate=sum(r.value_match for r in results) / n,
        mean_reward=sum(r.reward for r in results) / n,
        error_rate=errors / n,
        exact_match_ci=wilson_interval(exact, n),
        per_difficulty=per_difficulty,
    )


def summary_to_dict(summary: Summary) -> dict[str, Any]:
    return {
        "predictor": summary.predictor,
        "n": summary.n,
        "exact_match_rate": round(summary.exact_match_rate, 4),
        "result_match_rate": round(summary.result_match_rate, 4),
        "value_match_rate": round(summary.value_match_rate, 4),
        "mean_reward": round(summary.mean_reward, 4),
        "error_rate": round(summary.error_rate, 4),
        "exact_match_ci95": [round(summary.exact_match_ci[0], 4),
                             round(summary.exact_match_ci[1], 4)],
        "per_difficulty": {
            k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                for kk, vv in v.items()}
            for k, v in summary.per_difficulty.items()
        },
    }
