"""
HTTP client for the SQL Query Training Environment.

Provides a simple synchronous client for interacting with the running server.

Usage:
    from client import SQLEnvClient

    client = SQLEnvClient("http://localhost:8000")
    obs = client.reset(task_id="E1")
    print(obs["task_description"])

    obs = client.step("SELECT name, price FROM products ORDER BY id")
    print(obs["reward"], obs["feedback"])
"""

import json
from typing import Any
from urllib import error, request

from models import SQLAction, SQLObservation, SQLState


class SQLEnvClient:
    """Synchronous HTTP client for the SQL Query Training Environment."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _request_json(
        self, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None
    ) -> Any:
        url = f"{self._base_url}/{path.lstrip('/')}"
        body = None
        headers = {"Accept": "application/json"}

        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url, data=body, headers=headers, method=method)

        try:
            with request.urlopen(req, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"{method} {url} failed with status {exc.code}: {detail or exc.reason}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc

    def health(self) -> dict:
        return self._request_json("/health")

    def reset(self, task_id: str | None = None) -> SQLObservation:
        body = {"task_id": task_id} if task_id else {}
        return SQLObservation(**self._request_json("/reset", method="POST", payload=body))

    def step(self, sql_query: str) -> SQLObservation:
        action = SQLAction(sql_query=sql_query)
        return SQLObservation(
            **self._request_json("/step", method="POST", payload=action.model_dump())
        )

    def state(self) -> SQLState:
        return SQLState(**self._request_json("/state"))

    def tasks(self) -> list[dict]:
        return self._request_json("/tasks")

    def close(self) -> None:
        return None

    def __enter__(self) -> "SQLEnvClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
