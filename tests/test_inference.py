import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SCRIPT = ROOT / "inference.py"


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


if __name__ == "__main__":
    unittest.main()
