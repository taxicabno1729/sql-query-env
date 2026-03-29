"""
SQL execution and result-comparison grader.

Scoring rubric:
  1.0  – exact match (same rows, same columns, same values)
  0.7  – correct columns, ≥ 90 % of expected rows present
  0.4  – correct columns, 50–89 % of rows present
  0.2  – some correct columns but wrong row count / extras
  0.0  – SQL error or completely wrong result
"""

import sqlite3
from dataclasses import dataclass


@dataclass
class GradeResult:
    score: float
    feedback: str
    got_rows: int
    expected_rows: int
    got_columns: list[str]
    expected_columns: list[str]
    error: str | None = None


def _normalise_row(row: tuple) -> tuple:
    """Round floats to 2 dp for comparison; leave everything else as-is."""
    result = []
    for val in row:
        if isinstance(val, float):
            result.append(round(val, 2))
        else:
            result.append(val)
    return tuple(result)


def _column_overlap(got: list[str], expected: list[str]) -> float:
    got_set = {c.lower() for c in got}
    exp_set = {c.lower() for c in expected}
    if not exp_set:
        return 1.0
    return len(got_set & exp_set) / len(exp_set)


def grade(
    conn: sqlite3.Connection,
    agent_sql: str,
    reference_sql: str,
) -> GradeResult:
    """Execute both queries and compare results."""

    # Run reference query first to get expected output
    try:
        cur = conn.execute(reference_sql)
        expected_cols = [d[0] for d in cur.description]
        expected_rows = [_normalise_row(r) for r in cur.fetchall()]
    except sqlite3.Error as e:
        return GradeResult(
            score=0.0,
            feedback=f"Internal error running reference query: {e}",
            got_rows=0,
            expected_rows=0,
            got_columns=[],
            expected_columns=[],
            error=str(e),
        )

    # Run agent query
    try:
        cur = conn.execute(agent_sql)
        got_cols = [d[0] for d in cur.description]
        got_rows = [_normalise_row(r) for r in cur.fetchall()]
    except sqlite3.Error as e:
        return GradeResult(
            score=0.0,
            feedback=f"SQL error: {e}",
            got_rows=0,
            expected_rows=len(expected_rows),
            got_columns=[],
            expected_columns=expected_cols,
            error=str(e),
        )

    # ── Column check ──────────────────────────────────────────────────────
    col_overlap = _column_overlap(got_cols, expected_cols)
    cols_match = col_overlap >= 1.0

    # ── Row check ─────────────────────────────────────────────────────────
    n_expected = len(expected_rows)
    n_got = len(got_rows)

    if n_expected == 0 and n_got == 0:
        row_ratio = 1.0
    elif n_expected == 0:
        row_ratio = 0.0
    else:
        # Count how many expected rows appear in got (multiset intersection)
        got_counter: dict[tuple, int] = {}
        for r in got_rows:
            got_counter[r] = got_counter.get(r, 0) + 1

        matched = 0
        for r in expected_rows:
            if got_counter.get(r, 0) > 0:
                matched += 1
                got_counter[r] -= 1

        row_ratio = matched / n_expected

    # ── Score ─────────────────────────────────────────────────────────────
    if cols_match and row_ratio == 1.0 and n_got == n_expected:
        score = 1.0
        feedback = "Perfect match! All rows and columns are correct."
    elif cols_match and row_ratio >= 0.9:
        score = 0.7
        feedback = (
            f"Correct columns. Found {int(row_ratio * 100)}% of expected rows "
            f"({matched}/{n_expected}). Check for missing rows."
        )
    elif cols_match and row_ratio >= 0.5:
        score = 0.4
        feedback = (
            f"Correct columns but only {int(row_ratio * 100)}% of expected rows "
            f"({matched}/{n_expected}). Review your WHERE / JOIN conditions."
        )
    elif col_overlap >= 0.5:
        score = 0.2
        feedback = (
            f"Partial column match ({int(col_overlap * 100)}%). "
            f"Got {n_got} rows vs {n_expected} expected. "
            "Check column names and query logic."
        )
    else:
        score = 0.0
        feedback = (
            f"Wrong result. Got columns {got_cols} but expected {expected_cols}. "
            f"Got {n_got} rows vs {n_expected} expected."
        )

    return GradeResult(
        score=score,
        feedback=feedback,
        got_rows=n_got,
        expected_rows=n_expected,
        got_columns=got_cols,
        expected_columns=expected_cols,
    )
