"""
Core environment state machine for the SQL Query Training Environment.
"""

import sqlite3
from pathlib import Path

from .models import SQLAction, SQLObservation, SQLState
from .tasks.definitions import SCHEMA_INFO, TASK_MAP, TASKS, Task
from .tasks.grader import grade

_SCHEMA_PATH = Path(__file__).parent.parent / "data" / "schema.sql"


def _load_db() -> sqlite3.Connection:
    """Create a fresh in-memory SQLite database from schema.sql."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA_PATH.read_text())
    conn.commit()
    return conn


class SQLEnvironment:
    """
    Manages a single episode of the SQL query training environment.

    State transitions:
      idle  ──reset()──►  active  ──step()──►  done
                              ▲_______step() if not done___|
    """

    def __init__(self) -> None:
        self._conn: sqlite3.Connection | None = None
        self._task: Task | None = None
        self._attempts: int = 0
        self._score_history: list[float] = []
        self._task_index: int = 0  # cycles through TASKS in order

    # ── Public API ────────────────────────────────────────────────────────

    def reset(self, task_id: str | None = None) -> SQLObservation:
        """
        Start a new episode.  If task_id is None, cycle through tasks in order.
        """
        if task_id is not None:
            if task_id not in TASK_MAP:
                raise ValueError(f"Unknown task_id '{task_id}'. "
                                 f"Valid IDs: {list(TASK_MAP)}")
            self._task = TASK_MAP[task_id]
        else:
            self._task = TASKS[self._task_index % len(TASKS)]
            self._task_index += 1

        self._conn = _load_db()
        self._attempts = 0
        self._score_history = []

        return SQLObservation(
            done=False,
            reward=0.0,
            task_description=self._task.question,
            schema_info=SCHEMA_INFO,
            feedback="Episode started. Submit your SQL query.",
            metadata={
                "task_id": self._task.task_id,
                "difficulty": self._task.difficulty,
                "max_attempts": self._task.max_attempts,
            },
        )

    def step(self, action: SQLAction) -> SQLObservation:
        if self._task is None or self._conn is None:
            raise RuntimeError("Call reset() before step().")

        self._attempts += 1
        result = grade(self._conn, action.sql_query, self._task.reference_sql)
        self._score_history.append(result.score)

        done = (
            result.score == 1.0
            or self._attempts >= self._task.max_attempts
        )

        return SQLObservation(
            done=done,
            reward=result.score,
            task_description=self._task.question,
            schema_info=SCHEMA_INFO,
            feedback=result.feedback,
            metadata={
                "task_id": self._task.task_id,
                "difficulty": self._task.difficulty,
                "attempts": self._attempts,
                "max_attempts": self._task.max_attempts,
                "got_rows": result.got_rows,
                "expected_rows": result.expected_rows,
                "got_columns": result.got_columns,
                "expected_columns": result.expected_columns,
                "error": result.error,
                "score_history": list(self._score_history),
            },
        )

    def state(self) -> SQLState:
        if self._task is None:
            raise RuntimeError("Call reset() before state().")
        return SQLState(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            attempts=self._attempts,
            max_attempts=self._task.max_attempts,
            score_history=list(self._score_history),
        )
