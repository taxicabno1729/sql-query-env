import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SCRIPT = ROOT / "inference.py"


class TestHTTPServer:
    def __init__(self, handler_cls: type[BaseHTTPRequestHandler]) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class MockLLMHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/chat/completions":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        assert payload["model"] == "test-model"

        response = {
            "choices": [
                {
                    "message": {
                        "content": "```sql\nSELECT name, price FROM products ORDER BY id\n```"
                    }
                }
            ]
        }
        data = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


class MockEnvHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            payload = {"status": "healthy"}
        elif self.path == "/tasks":
            payload = [{"task_id": "E1"}]
        else:
            self.send_error(404)
            return

        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))

        if self.path == "/reset":
            assert payload == {"task_id": "E1"}
            response = {
                "done": False,
                "reward": 0.0,
                "task_description": "List product names and prices.",
                "schema_info": "products(id, name, price)",
                "feedback": "Episode started. Submit your SQL query.",
            }
        elif self.path == "/step":
            assert "SELECT name, price FROM products ORDER BY id" in payload["sql_query"]
            response = {
                "done": True,
                "reward": 1.0,
                "task_description": "List product names and prices.",
                "schema_info": "products(id, name, price)",
                "feedback": "Exact match.",
            }
        else:
            self.send_error(404)
            return

        data = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


class BadHealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            payload = {"status": "starting"}
        else:
            self.send_error(404)
            return

        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


class InferenceScriptTests(unittest.TestCase):
    def run_inference(self, env_overrides: dict[str, str | None]) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.pop("HF_TOKEN", None)
        env.pop("API_BASE_URL", None)
        env.pop("MODEL_NAME", None)
        env.pop("ENV_URL", None)
        env.update({k: v for k, v in env_overrides.items() if v is not None})

        return subprocess.run(
            [sys.executable, str(INFERENCE_SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

    def assert_json_error(self, output: str) -> dict:
        payload = json.loads(output.strip())
        self.assertEqual(payload["type"], "ERROR")
        self.assertIn("message", payload)
        return payload

    def parse_json_lines(self, output: str) -> list[dict]:
        return [json.loads(line) for line in output.splitlines() if line.strip()]

    def test_reports_missing_hf_token_without_traceback(self) -> None:
        result = self.run_inference({})

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "")
        payload = self.assert_json_error(result.stdout)
        self.assertIn("HF_TOKEN env var is required", payload["message"])

    def test_reports_unreachable_environment_without_traceback(self) -> None:
        result = self.run_inference(
            {
                "HF_TOKEN": "dummy-token",
                "ENV_URL": "http://127.0.0.1:9",
            }
        )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "")
        payload = self.assert_json_error(result.stdout)
        self.assertIn("Cannot reach environment", payload["message"])

    def test_runs_locally_against_mock_env_and_model(self) -> None:
        env_server = TestHTTPServer(MockEnvHandler)
        llm_server = TestHTTPServer(MockLLMHandler)
        env_server.start()
        llm_server.start()
        self.addCleanup(env_server.stop)
        self.addCleanup(llm_server.stop)

        result = self.run_inference(
            {
                "HF_TOKEN": "dummy-token",
                "API_BASE_URL": llm_server.base_url,
                "MODEL_NAME": "test-model",
                "ENV_URL": env_server.base_url,
            }
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "")
        events = self.parse_json_lines(result.stdout)
        self.assertEqual([event["type"] for event in events], ["START", "STEP", "END", "SUMMARY"])
        self.assertEqual(events[0]["task_id"], "E1")
        self.assertEqual(events[1]["reward"], 1.0)
        self.assertEqual(events[2]["attempts"], 1)
        self.assertEqual(events[3]["average_score"], 1.0)

    def test_reports_unhealthy_environment_status_without_traceback(self) -> None:
        env_server = TestHTTPServer(BadHealthHandler)
        env_server.start()
        self.addCleanup(env_server.stop)

        result = self.run_inference(
            {
                "HF_TOKEN": "dummy-token",
                "ENV_URL": env_server.base_url,
            }
        )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "")
        payload = self.assert_json_error(result.stdout)
        self.assertIn("Cannot reach environment", payload["message"])
        self.assertIn("unexpected health status", payload["message"])


if __name__ == "__main__":
    unittest.main()
