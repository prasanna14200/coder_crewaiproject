import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from coder import web


class CodingWebTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web.app)

    def test_home_has_task_input_output_and_download(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('name="assignment"', response.text)
        self.assertIn('id="output"', response.text)
        self.assertIn('id="download"', response.text)
        self.assertIn("not executed", response.text)

    @patch.object(web, "Coder")
    @patch.object(web, "executor")
    def test_job_returns_generated_result_without_execution(self, executor, coder_class):
        crew = coder_class.return_value.crew.return_value
        crew.tasks = [SimpleNamespace(output_file="output/code_and_output.txt")]
        crew.kickoff.return_value = SimpleNamespace(
            raw="def is_palindrome(value):\n    return value == value[::-1]",
            tasks_output=[],
        )
        executor.submit.side_effect = lambda function, *args: function(*args)

        started = self.client.post(
            "/jobs",
            json={"assignment": "Write a Python palindrome function and tests."},
        )
        self.assertEqual(started.status_code, 202)
        result = self.client.get(f"/jobs/{started.json()['job_id']}")

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "completed")
        self.assertIn("is_palindrome", result.json()["result"])
        coder_class.assert_called_once_with(allow_code_execution=False)
        self.assertIsNone(crew.tasks[0].output_file)
        submitted_assignment = crew.kickoff.call_args.kwargs["inputs"]["assignment"]
        self.assertIn("Do not execute generated code", submitted_assignment)

    def test_job_rejects_short_assignment(self):
        response = self.client.post("/jobs", json={"assignment": "too short"})

        self.assertEqual(response.status_code, 422)

    def test_health_check(self):
        response = self.client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_non_executing_crew_does_not_require_docker(self):
        crew = web.Coder(allow_code_execution=False).crew()

        self.assertFalse(crew.agents[0].allow_code_execution)


if __name__ == "__main__":
    unittest.main()