"""Tests for the evaluation harness.

These are the guardrails on the published numbers: if the oracle stops scoring
1.0, or the naive floor starts scoring well, the metrics are wrong.
"""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from server.tasks.definitions import SCHEMA_INFO

from eval.dataset import GOLDEN_SET
from eval.metrics import evaluate_item, summarise, wilson_interval
from eval.predictors import NaivePredictor, OraclePredictor, extract_sql
from eval.run import fresh_connection

ROOT = Path(__file__).resolve().parents[1]


def _result(item, sql: str, conn: sqlite3.Connection):
    return evaluate_item(
        conn,
        item_id=item.id,
        difficulty=item.difficulty,
        question=item.question,
        predicted_sql=sql,
        reference_sql=item.reference_sql,
    )


class TestGoldenSet(unittest.TestCase):
    def test_ids_unique(self):
        ids = [item.id for item in GOLDEN_SET]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_reference_query_runs_and_returns_rows(self):
        conn = fresh_connection()
        for item in GOLDEN_SET:
            with self.subTest(item=item.id):
                cur = conn.execute(item.reference_sql)
                self.assertTrue(cur.fetchall(), f"{item.id} returned no rows")

    def test_all_difficulties_present(self):
        difficulties = {item.difficulty for item in GOLDEN_SET}
        self.assertEqual(difficulties, {"easy", "medium", "hard", "expert"})


class TestOracle(unittest.TestCase):
    """A correct answer must score 1.0 — otherwise the harness is broken."""

    def test_oracle_scores_perfect(self):
        conn = fresh_connection()
        oracle = OraclePredictor()
        results = [
            _result(item, oracle.predict(item, SCHEMA_INFO), conn) for item in GOLDEN_SET
        ]
        self.assertTrue(all(r.reward == 1.0 for r in results))
        summary = summarise("oracle", results)
        self.assertEqual(summary.exact_match_rate, 1.0)
        self.assertEqual(summary.error_rate, 0.0)


class TestNaiveFloor(unittest.TestCase):
    def test_naive_only_matches_the_single_table_item(self):
        conn = fresh_connection()
        naive = NaivePredictor()
        results = [
            _result(item, naive.predict(item, SCHEMA_INFO), conn) for item in GOLDEN_SET
        ]
        matched = [r.item_id for r in results if r.exact_match]
        self.assertEqual(matched, ["E01"])
        summary = summarise("naive", results)
        self.assertAlmostEqual(summary.exact_match_rate, 1 / len(GOLDEN_SET))


class TestScoringEdgeCases(unittest.TestCase):
    def setUp(self):
        self.conn = fresh_connection()
        self.item = GOLDEN_SET[0]  # E01: name, price ordered by id

    def test_sql_error_is_recorded_not_raised(self):
        result = _result(self.item, "SELECT nope FROM missing_table", self.conn)
        self.assertFalse(result.exact_match)
        self.assertEqual(result.reward, 0.0)
        self.assertIsNotNone(result.error)

    def test_wrong_row_order_is_not_exact_but_is_result_match(self):
        # The environment grader is order-insensitive (same rows, same columns
        # => 1.0), while exact_match is deliberately stricter. Both facts are
        # asserted here so the difference can never drift silently.
        shuffled = "SELECT name, price FROM products ORDER BY id DESC"
        result = _result(self.item, shuffled, self.conn)
        self.assertFalse(result.exact_match)
        self.assertTrue(result.result_match)
        self.assertEqual(result.reward, 1.0)

    def test_subset_is_partial_not_exact(self):
        partial = "SELECT name, price FROM products WHERE id <= 10 ORDER BY id"
        result = _result(self.item, partial, self.conn)
        self.assertFalse(result.exact_match)
        self.assertGreater(result.reward, 0.0)
        self.assertLess(result.reward, 1.0)


class TestValueMatch(unittest.TestCase):
    """Alias differences are style, not wrong answers — a separate metric."""

    def test_different_alias_still_value_matches(self):
        conn = fresh_connection()
        item = GOLDEN_SET[2]  # E03: SELECT COUNT(*) AS order_count FROM orders
        result = _result(item, "SELECT COUNT(*) FROM orders", conn)
        self.assertFalse(result.exact_match)
        self.assertFalse(result.result_match)
        self.assertTrue(result.value_match)

    def test_wrong_values_do_not_value_match(self):
        conn = fresh_connection()
        item = GOLDEN_SET[2]
        result = _result(item, "SELECT COUNT(*) + 1 FROM orders", conn)
        self.assertFalse(result.value_match)


class TestWilsonInterval(unittest.TestCase):
    def test_bounds(self):
        lo, hi = wilson_interval(12, 12)
        self.assertGreater(lo, 0.7)
        self.assertAlmostEqual(hi, 1.0, places=9)

    def test_zero_successes_has_positive_upper_bound(self):
        lo, hi = wilson_interval(0, 12)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        self.assertLess(hi, 0.3)

    def test_empty_sample(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))


class TestExtractSql(unittest.TestCase):
    def test_fenced_block(self):
        reply = "Sure!\n```sql\nSELECT 1;\n```\n"
        self.assertEqual(extract_sql(reply), "SELECT 1")

    def test_bare_query(self):
        self.assertEqual(extract_sql("SELECT 2"), "SELECT 2")


if __name__ == "__main__":
    unittest.main()
