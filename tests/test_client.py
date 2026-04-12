import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


class MockEnvHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            payload = {"status": "healthy"}
        elif self.path == "/tasks":
            payload = [{"task_id": "E1"}]
        elif self.path == "/state":
            payload = {
                "current_task": "E1",
                "attempts": 0,
                "max_attempts": 5,
                "score_history": [],
                "done": False,
            }
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
        payload = json.loads(self.rfile.read(length).decode("utf-8"))

        if self.path == "/reset":
            assert payload == {"task_id": "E1"}
            response = {
                "done": False,
                "reward": 0.0,
                "task_description": "List product names and prices.",
                "schema_info": "products(id, name, price)",
                "feedback": "Episode started. Submit your SQL query.",
                "metadata": {"task_id": "E1", "max_attempts": 5},
            }
        elif self.path == "/step":
            assert payload == {"sql_query": "SELECT 1"}
            response = {
                "done": True,
                "reward": 1.0,
                "task_description": "List product names and prices.",
                "schema_info": "products(id, name, price)",
                "feedback": "Exact match.",
                "metadata": {"attempts": 1, "max_attempts": 5},
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


class ClientTests(unittest.TestCase):
    def test_client_imports_and_works_without_httpx_installed(self) -> None:
        server = TestHTTPServer(MockEnvHandler)
        server.start()
        self.addCleanup(server.stop)

        with tempfile.TemporaryDirectory() as tmpdir:
            sitecustomize = Path(tmpdir) / "sitecustomize.py"
            sitecustomize.write_text(
                textwrap.dedent(
                    """
                    import builtins

                    _orig_import = builtins.__import__

                    def guarded_import(name, *args, **kwargs):
                        if name == "httpx":
                            raise ModuleNotFoundError("No module named 'httpx'")
                        return _orig_import(name, *args, **kwargs)

                    builtins.__import__ = guarded_import
                    """
                ),
                encoding="utf-8",
            )
            models_stub = Path(tmpdir) / "models.py"
            models_stub.write_text(
                textwrap.dedent(
                    """
                    class SQLAction:
                        def __init__(self, sql_query):
                            self.sql_query = sql_query

                        def model_dump(self):
                            return {"sql_query": self.sql_query}

                    class SQLObservation:
                        def __init__(self, **kwargs):
                            self.__dict__.update(kwargs)

                    class SQLState:
                        def __init__(self, **kwargs):
                            self.__dict__.update(kwargs)
                    """
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)

            code = textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {tmpdir!r})

                from client import SQLEnvClient

                with SQLEnvClient({server.base_url!r}) as client:
                    print(client.health()["status"])
                    print(client.tasks()[0]["task_id"])
                    print(client.reset(task_id="E1").metadata["task_id"])
                    print(client.step("SELECT 1").reward)
                    print(client.state().current_task)
                """
            )

            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=tmpdir,
                env=env,
                text=True,
                capture_output=True,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(
            result.stdout.strip().splitlines(),
            ["healthy", "E1", "E1", "1.0", "E1"],
        )


if __name__ == "__main__":
    unittest.main()
