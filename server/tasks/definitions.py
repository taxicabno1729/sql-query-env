"""
Task definitions for the SQL Query Training Environment.
Each task specifies a natural-language question and a reference SQL query.
"""

from dataclasses import dataclass, field


@dataclass
class Task:
    task_id: str
    difficulty: str  # easy | medium | hard
    question: str
    schema_hint: str
    reference_sql: str
    max_attempts: int = 5
    tags: list[str] = field(default_factory=list)


SCHEMA_INFO = """
Tables available:
  customers(id, name, email, city)
  products(id, name, category, price)
  orders(id, customer_id, order_date, status)
  order_items(id, order_id, product_id, quantity, unit_price)

Relationships:
  orders.customer_id -> customers.id
  order_items.order_id -> orders.id
  order_items.product_id -> products.id
"""

TASKS: list[Task] = [
    # ── Easy tasks ─────────────────────────────────────────────────────────
    Task(
        task_id="E1",
        difficulty="easy",
        question="List all product names and their prices.",
        schema_hint="Use the products table. Return name and price columns.",
        reference_sql="SELECT name, price FROM products ORDER BY id",
        tags=["select", "basic"],
    ),
    Task(
        task_id="E2",
        difficulty="easy",
        question="Find all customers who live in New York. Return their names and emails.",
        schema_hint="Use the customers table. Filter by city.",
        reference_sql=(
            "SELECT name, email FROM customers "
            "WHERE city = 'New York' ORDER BY id"
        ),
        tags=["select", "filter"],
    ),
    Task(
        task_id="E3",
        difficulty="easy",
        question="How many orders were placed in total?",
        schema_hint="Use the orders table. Use COUNT(*).",
        reference_sql="SELECT COUNT(*) AS total_orders FROM orders",
        tags=["aggregate", "count"],
    ),
    # ── Medium tasks ───────────────────────────────────────────────────────
    Task(
        task_id="M1",
        difficulty="medium",
        question=(
            "Find the total revenue per product category. "
            "Return category and total_revenue, ordered by total_revenue descending."
        ),
        schema_hint=(
            "Join products with order_items. "
            "Multiply quantity * unit_price to get revenue. Group by category."
        ),
        reference_sql=(
            "SELECT p.category, "
            "       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue "
            "FROM products p "
            "JOIN order_items oi ON p.id = oi.product_id "
            "GROUP BY p.category "
            "ORDER BY total_revenue DESC"
        ),
        tags=["join", "aggregate", "group-by"],
    ),
    Task(
        task_id="M2",
        difficulty="medium",
        question=(
            "List the top 5 customers by total amount spent. "
            "Return customer name and total_spent, ordered by total_spent descending."
        ),
        schema_hint=(
            "Join customers -> orders -> order_items. "
            "Sum quantity * unit_price per customer. Use LIMIT 5."
        ),
        reference_sql=(
            "SELECT c.name, "
            "       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent "
            "FROM customers c "
            "JOIN orders o ON c.id = o.customer_id "
            "JOIN order_items oi ON o.id = oi.order_id "
            "GROUP BY c.id, c.name "
            "ORDER BY total_spent DESC "
            "LIMIT 5"
        ),
        tags=["join", "aggregate", "limit"],
    ),
    Task(
        task_id="M3",
        difficulty="medium",
        question=(
            "Find all distinct orders that include at least one product from "
            "the 'Electronics' category. Return the order id and order_date."
        ),
        schema_hint=(
            "Join orders -> order_items -> products. "
            "Filter where category = 'Electronics'. Use DISTINCT."
        ),
        reference_sql=(
            "SELECT DISTINCT o.id, o.order_date "
            "FROM orders o "
            "JOIN order_items oi ON o.id = oi.order_id "
            "JOIN products p ON oi.product_id = p.id "
            "WHERE p.category = 'Electronics' "
            "ORDER BY o.id"
        ),
        tags=["join", "filter", "distinct"],
    ),
    # ── Hard tasks ─────────────────────────────────────────────────────────
    Task(
        task_id="H1",
        difficulty="hard",
        question=(
            "Rank customers by total spend within each city using window functions. "
            "Return city, customer name, total_spent, and city_rank "
            "(1 = highest spender in that city). "
            "Order results by city, then city_rank."
        ),
        schema_hint=(
            "Use RANK() OVER (PARTITION BY city ORDER BY total_spent DESC). "
            "You may need a CTE or subquery to compute total_spent first."
        ),
        reference_sql=(
            "WITH customer_spend AS ("
            "  SELECT c.id, c.name, c.city, "
            "         ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent "
            "  FROM customers c "
            "  JOIN orders o ON c.id = o.customer_id "
            "  JOIN order_items oi ON o.id = oi.order_id "
            "  GROUP BY c.id, c.name, c.city"
            ") "
            "SELECT city, name, total_spent, "
            "       RANK() OVER (PARTITION BY city ORDER BY total_spent DESC) AS city_rank "
            "FROM customer_spend "
            "ORDER BY city, city_rank"
        ),
        max_attempts=7,
        tags=["window-function", "cte", "rank"],
    ),
    Task(
        task_id="H2",
        difficulty="hard",
        question=(
            "Find customers whose total spending is above the average customer spend. "
            "Return their name, total_spent, and their percentile rank (0.0–1.0) "
            "among all customers who made purchases, ordered by total_spent descending."
        ),
        schema_hint=(
            "Use a CTE to compute per-customer totals. "
            "Use PERCENT_RANK() OVER (ORDER BY total_spent) for percentile. "
            "Filter where total_spent > AVG(total_spent)."
        ),
        reference_sql=(
            "WITH customer_spend AS ("
            "  SELECT c.name, "
            "         ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent "
            "  FROM customers c "
            "  JOIN orders o ON c.id = o.customer_id "
            "  JOIN order_items oi ON o.id = oi.order_id "
            "  GROUP BY c.id, c.name"
            "), "
            "ranked AS ("
            "  SELECT name, total_spent, "
            "         ROUND(PERCENT_RANK() OVER (ORDER BY total_spent), 4) AS pct_rank "
            "  FROM customer_spend"
            ") "
            "SELECT name, total_spent, pct_rank "
            "FROM ranked "
            "WHERE total_spent > (SELECT AVG(total_spent) FROM customer_spend) "
            "ORDER BY total_spent DESC"
        ),
        max_attempts=7,
        tags=["window-function", "cte", "percent-rank"],
    ),
]

TASK_MAP: dict[str, Task] = {t.task_id: t for t in TASKS}
