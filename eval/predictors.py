"""
Predictors for the text-to-SQL evaluation harness.

A predictor is anything that turns ``(question, schema)`` into a SQL string.
Three are shipped:

``oracle``   returns the item's reference SQL. It exists to prove the harness
             itself is sound: a correct answer must score 1.0 on every item.
``naive``    a fixed, deterministic template. It is the floor — the score a
             non-reasoning baseline gets, with no network and no model.
``llm``      an OpenAI-compatible chat completion (OpenRouter, vLLM, Ollama,
             OpenAI, anything speaking /chat/completions). Requires
             ``EVAL_LLM_API_KEY``; nothing else in the harness does.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .dataset import EvalItem

PROMPT_TEMPLATE = """You write SQLite queries.

Schema:
{schema}

Write exactly one SQLite SELECT statement that answers the question.
Return only the query, inside a ```sql code block. No explanation.

Question: {question}
"""

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)


class Predictor(Protocol):
    name: str

    def predict(self, item: EvalItem, schema: str) -> str: ...


@dataclass
class OraclePredictor:
    """Returns the reference query. Sanity check for the harness, not a model."""

    name: str = "oracle"

    def predict(self, item: EvalItem, schema: str) -> str:
        return item.reference_sql


@dataclass
class NaivePredictor:
    """
    Deterministic floor baseline: always the same plausible-looking query.

    Deliberately dumb. Its job is to keep the harness honest — if a change makes
    the naive baseline look good, the scoring is broken.
    """

    name: str = "naive"

    def predict(self, item: EvalItem, schema: str) -> str:
        return "SELECT name, price FROM products ORDER BY id"


@dataclass
class LLMPredictor:
    """
    OpenAI-compatible chat-completion predictor.

    Configuration comes from the environment so no key ever lands in the repo:

        EVAL_LLM_API_KEY   required
        EVAL_LLM_BASE_URL  default https://openrouter.ai/api/v1
        EVAL_LLM_MODEL     default anthropic/claude-sonnet-4.5
    """

    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    timeout_s: int = 60
    name: str = "llm"

    def __post_init__(self) -> None:
        self.model = self.model or os.getenv("EVAL_LLM_MODEL", "anthropic/claude-sonnet-4.5")
        self.base_url = (
            self.base_url or os.getenv("EVAL_LLM_BASE_URL", "https://openrouter.ai/api/v1")
        ).rstrip("/")
        self.api_key = self.api_key or os.getenv("EVAL_LLM_API_KEY")
        if not self.api_key:
            raise SystemExit(
                "EVAL_LLM_API_KEY is not set — run with --predictor oracle or naive, "
                "or export a key for the llm predictor."
            )

    def predict(self, item: EvalItem, schema: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": 400,
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(
                            schema=schema, question=item.question
                        ),
                    }
                ],
            }
        ).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"model call failed: HTTP {exc.code} {exc.read()[:200]!r}"
            ) from exc

        text = payload["choices"][0]["message"]["content"]
        return extract_sql(text)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply; tolerate prose around the block."""
    match = _SQL_BLOCK.search(text)
    candidate = match.group(1) if match else text
    candidate = candidate.strip().rstrip(";").strip()
    return candidate


def build_predictor(kind: str, model: str | None = None) -> Predictor:
    if kind == "oracle":
        return OraclePredictor()
    if kind == "naive":
        return NaivePredictor()
    if kind == "llm":
        predictor = LLMPredictor(model=model)
        predictor.name = f"llm:{predictor.model}"
        return predictor
    raise SystemExit(f"unknown predictor {kind!r} (choose oracle, naive or llm)")
