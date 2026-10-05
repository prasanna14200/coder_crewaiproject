import os
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from coder import model_check, web


class CodingWebTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web.app)
        with web.jobs_lock:
            web.jobs.clear()
            web.active_job = None
            web.daily_job_date = None
            web.daily_job_count = 0

    def test_home_has_task_input_output_download_and_free_model_notice(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('name="assignment"', response.text)
        self.assertIn('maxlength="1200"', response.text)
        self.assertIn('id="output"', response.text)
        self.assertIn('id="download"', response.text)
        self.assertIn("Gemini 3.8 Flash / Free tier", response.text)
        self.assertIn("not executed", response.text)

    @patch.object(web, "Coder")
    @patch.object(web, "executor")
    def test_job_returns_generated_result_without_execution(self, executor, coder_class):
        crew = coder_class.return_value.crew.return_value
        crew.tasks = [SimpleNamespace(output_file="output/code_and_output.txt")]
        crew.kickoff.return_value = SimpleNamespace(
            raw="def grade_marks(marks):\n    total = sum(marks)\n    return total, total / len(marks), 'B'",
            tasks_output=[],
        )
        executor.submit.side_effect = lambda function, *args: function(*args)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            started = self.client.post(
                "/jobs",
                json={"assignment": "Calculate total, average, and grade for marks 90, 80, 70, 85, 95."},
            )
        result = self.client.get(f"/jobs/{started.json()['job_id']}")

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "completed")
        self.assertIn("grade_marks", result.json()["result"])
        coder_class.assert_called_once_with(allow_code_execution=False)
        self.assertIsNone(crew.tasks[0].output_file)

    def test_job_rejects_short_assignment(self):
        response = self.client.post("/jobs", json={"assignment": "too short"})

        self.assertEqual(response.status_code, 422)

    def test_job_reports_missing_gemini_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            response = self.client.post(
                "/jobs",
                json={"assignment": "Create a Python program for calculating grades."},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("GEMINI_API_KEY", response.json()["detail"])

    @patch.object(web, "Coder")
    @patch.object(web, "executor")
    def test_permanent_model_error_is_safe_and_not_retried(self, executor, coder_class):
        crew = coder_class.return_value.crew.return_value
        crew.tasks = [SimpleNamespace(output_file="output/code_and_output.txt")]
        crew.kickoff.side_effect = RuntimeError("model_not_found: private provider response")
        executor.submit.side_effect = lambda function, *args: function(*args)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with self.assertLogs(web.logger, level="ERROR") as captured:
                started = self.client.post(
                    "/jobs",
                    json={"assignment": "Create a Python program for calculating grades."},
                )
        result = self.client.get(f"/jobs/{started.json()['job_id']}").json()

        self.assertEqual(result["status"], "failed")
        self.assertIn("pinned to gemini-3.8-flash", result["detail"])
        self.assertNotIn("private provider response", result["detail"])
        self.assertTrue(any("model_not_found" in line for line in captured.output))
        crew.kickoff.assert_called_once()

    @patch.object(web, "Coder")
    @patch.object(web, "executor")
    @patch.object(web.time, "sleep")
    def test_rate_limit_gets_only_two_bounded_retries(self, sleep, executor, coder_class):
        crew = coder_class.return_value.crew.return_value
        crew.tasks = [SimpleNamespace(output_file="output/code_and_output.txt")]

        class TooManyRequests(Exception):
            status_code = 429

        crew.kickoff.side_effect = [
            TooManyRequests("temporary rate limit"),
            TooManyRequests("temporary rate limit"),
            TooManyRequests("temporary rate limit"),
        ]
        executor.submit.side_effect = lambda function, *args: function(*args)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            started = self.client.post(
                "/jobs",
                json={"assignment": "Calculate total, average, and grade for 5 marks."},
            )
        result = self.client.get(f"/jobs/{started.json()['job_id']}").json()

        self.assertEqual(result["status"], "failed")
        self.assertIn("free-tier rate limit", result["detail"])
        self.assertEqual(crew.kickoff.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    def test_daily_budget_rejects_extra_tasks(self):
        web.daily_job_date = datetime.now(timezone.utc).date()
        web.daily_job_count = web.MAX_DAILY_JOBS

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            response = self.client.post(
                "/jobs",
                json={"assignment": "Calculate total, average, and grade for 5 marks."},
            )

        self.assertEqual(response.status_code, 429)
        self.assertIn("10-task daily", response.json()["detail"])

    @patch.object(model_check, "LLM")
    def test_direct_model_check_is_one_capped_call(self, llm_class):
        llm_class.return_value.call.return_value = "direct Gemini request succeeded"

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            response = model_check.request_model()

        self.assertEqual(response, "direct Gemini request succeeded")
        llm_class.assert_called_once_with(
            model="gemini/gemini-3.8-flash",
            api_key="test-key",
            timeout=20,
            max_tokens=48,
            num_retries=0,
        )

    @patch.object(model_check, "LLM")
    def test_direct_model_check_requires_key_before_request(self, llm_class):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                model_check.request_model()
        llm_class.assert_not_called()

    @patch.object(model_check, "LLM")
    def test_direct_cli_reports_missing_key_without_network(self, llm_class):
        output = StringIO()
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}), redirect_stdout(output):
            result = model_check.main()

        self.assertEqual(result, 1)
        self.assertIn("GEMINI_API_KEY is not configured", output.getvalue())
        llm_class.assert_not_called()

    def test_health_check(self):
        response = self.client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_non_executing_crew_does_not_require_docker(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            crew = web.Coder(allow_code_execution=False).crew()

        self.assertFalse(crew.agents[0].allow_code_execution)
        self.assertEqual(crew.agents[0].max_retry_limit, 0)


if __name__ == "__main__":
    unittest.main()