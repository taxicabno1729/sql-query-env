"""
Golden set for the text-to-SQL evaluation harness.

Each item is a natural-language question plus a reference SQL query that is
executed against the seeded e-commerce database (``data/schema.sql``). The
reference query defines the expected result, so an item is only valid if the
query runs and returns a non-empty result.

Items are deliberately spread over four difficulty bands so the harness can
report *where* a model fails, not just an aggregate score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Difficulty = Literal["easy", "medium", "hard", "expert"]


@dataclass(frozen=True)
class EvalItem:
    id: str
    difficulty: Difficulty
    question: str
    reference_sql: str
    notes: str = ""


GOLDEN_SET: tuple[EvalItem, ...] = (
    EvalItem(
        id="E01",
        difficulty="easy",
        question="List every product name with its price.",
        reference_sql="SELECT name, price FROM products ORDER BY id",
        notes="Single table, no filtering.",
    ),
    EvalItem(
        id="E02",
        difficulty="easy",
        question="Which customers live in New York? Give their names and emails.",
        reference_sql=(
            "SELECT name, email FROM customers "
            "WHERE city = 'New York' ORDER BY id"
        ),
    ),
    EvalItem(
        id="E03",
        difficulty="easy",
        question="How many orders are there in total?",
        reference_sql="SELECT COUNT(*) AS order_count FROM orders",
        notes="Aggregate without grouping.",
    ),
    EvalItem(
        id="E04",
        difficulty="easy",
        question="List the distinct cities that customers live in, sorted alphabetically.",
        reference_sql="SELECT DISTINCT city FROM customers ORDER BY city",
    ),
    EvalItem(
        id="M01",
        difficulty="medium",
        question=(
            "What is the total revenue for each product category? "
            "Revenue is quantity times unit price on the order line."
        ),
        reference_sql=(
            "SELECT p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
            "FROM order_items oi "
            "JOIN products p ON p.id = oi.product_id "
            "GROUP BY p.category ORDER BY p.category"
        ),
        notes="Two-table join plus aggregation.",
    ),
    EvalItem(
        id="M02",
        difficulty="medium",
        question="Who are the top 5 customers by total spend? Show name and amount.",
        reference_sql=(
            "SELECT c.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spend "
            "FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id "
            "GROUP BY c.id ORDER BY total_spend DESC, c.id LIMIT 5"
        ),
        notes="Three-table join, ordering and a limit.",
    ),
    EvalItem(
        id="M03",
        difficulty="medium",
        question=(
            "Which orders contain at least one Electronics product? "
            "Return each order id once, ascending."
        ),
        reference_sql=(
            "SELECT DISTINCT o.id FROM orders o "
            "JOIN order_items oi ON oi.order_id = o.id "
            "JOIN products p ON p.id = oi.product_id "
            "WHERE p.category = 'Electronics' ORDER BY o.id"
        ),
        notes="DISTINCT is the trap: a plain join duplicates orders.",
    ),
    EvalItem(
        id="M04",
        difficulty="medium",
        question="How many delivered orders were placed in March 2024?",
        reference_sql=(
            "SELECT COUNT(*) AS n FROM orders "
            "WHERE status = 'delivered' "
            "AND order_date >= '2024-03-01' AND order_date < '2024-04-01'"
        ),
        notes="Date range plus status filter.",
    ),
    EvalItem(
        id="H01",
        difficulty="hard",
        question=(
            "Rank customers by total spend within their own city. "
            "Return city, name, total spend and the rank in that city."
        ),
        reference_sql=(
            "SELECT city, name, total_spend, "
            "RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS city_rank "
            "FROM ("
            "  SELECT c.city AS city, c.name AS name, "
            "         SUM(oi.quantity * oi.unit_price) AS total_spend "
            "  FROM customers c "
            "  JOIN orders o ON o.customer_id = c.id "
            "  JOIN order_items oi ON oi.order_id = o.id "
            "  GROUP BY c.id"
            ") ORDER BY city, city_rank"
        ),
        notes="Window function over a grouped subquery.",
    ),
    EvalItem(
        id="H02",
        difficulty="hard",
        question=(
            "Which customers spend more than the average customer's total spend? "
            "Show name, total spend and the percentile rank of that spend."
        ),
        reference_sql=(
            "WITH spend AS ("
            "  SELECT c.id AS customer_id, c.name AS name, "
            "         SUM(oi.quantity * oi.unit_price) AS total_spend "
            "  FROM customers c "
            "  JOIN orders o ON o.customer_id = c.id "
            "  JOIN order_items oi ON oi.order_id = o.id "
            "  GROUP BY c.id"
            ") "
            "SELECT name, total_spend, "
            "PERCENT_RANK() OVER (ORDER BY total_spend) AS pct_rank "
            "FROM spend "
            "WHERE total_spend > (SELECT AVG(total_spend) FROM spend) "
            "ORDER BY total_spend DESC"
        ),
        notes="CTE, window function and correlated aggregate.",
    ),
    EvalItem(
        id="X01",
        difficulty="expert",
        question=(
            "For each category, show the most expensive product and its price."
        ),
        reference_sql=(
            "SELECT category, name, price FROM ("
            "  SELECT category, name, price, "
            "         ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC, id) AS rn "
            "  FROM products"
            ") WHERE rn = 1 ORDER BY category"
        ),
        notes="ROW_NUMBER per partition, then filter.",
    ),
    EvalItem(
        id="X02",
        difficulty="expert",
        question=(
            "What share of orders that contain at least one product were cancelled? "
            "Return the percentage rounded to two decimals."
        ),
        reference_sql=(
            "SELECT ROUND(100.0 * SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) "
            "/ COUNT(*), 2) AS cancelled_pct FROM orders"
        ),
        notes="Ratio with CASE and rounding.",
    ),
)


GOLDEN_BY_ID = {item.id: item for item in GOLDEN_SET}
