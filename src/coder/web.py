import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from coder.crew import Coder


logger = logging.getLogger(__name__)
app = FastAPI(title="CrewAI Code Writer")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coding-task")
jobs: dict[str, dict[str, str]] = {}
jobs_lock = threading.Lock()
active_job: str | None = None


class CodingRequest(BaseModel):
    assignment: str = Field(min_length=10, max_length=2000)


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f5f7f4">
  <title>CrewAI Code Writer</title>
  <style>
    :root { color-scheme: light; --ink: #172b2a; --muted: #63716d; --line: #d9e1dc; --paper: #f5f7f4; --white: #fff; --green: #176b57; --orange: #a54d30; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.55 system-ui, sans-serif; }
    header { background: var(--white); border-bottom: 1px solid var(--line); }
    .topline, main { width: min(100% - 40px, 960px); margin-inline: auto; }
    .topline { min-height: 64px; display: flex; align-items: center; justify-content: space-between; gap: 20px; }
    .brand { color: var(--ink); font-size: 14px; font-weight: 750; letter-spacing: .08em; text-decoration: none; text-transform: uppercase; }
    .model { color: var(--muted); font-size: 12px; text-align: right; }
    main { padding-block: 52px 80px; }
    h1 { max-width: 700px; margin: 0; font: 500 clamp(36px, 6vw, 60px)/1.04 Georgia, serif; letter-spacing: 0; }
    .intro { color: var(--muted); margin: 14px 0 30px; }
    form { padding: 20px; background: var(--white); border: 1px solid var(--line); border-top: 3px solid var(--green); }
    label { display: block; margin-bottom: 10px; font-size: 13px; font-weight: 700; }
    textarea { display: block; width: 100%; min-height: 140px; padding: 14px; border: 1px solid var(--line); border-radius: 3px; resize: vertical; color: var(--ink); font: inherit; }
    textarea:focus { outline: 2px solid var(--green); outline-offset: 2px; }
    .form-footer { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-top: 14px; }
    button, .download { display: inline-flex; min-height: 44px; align-items: center; justify-content: center; padding: 0 18px; border: 0; border-radius: 3px; background: var(--green); color: white; font: inherit; font-weight: 700; text-decoration: none; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    #status { min-height: 24px; color: var(--muted); font-size: 13px; }
    #status.error { color: #a3362d; }
    .notice { margin: 14px 0 0; padding: 12px 14px; border-left: 3px solid var(--orange); background: #f5ece6; color: #573729; font-size: 13px; }
    #result { margin-top: 38px; }
    .result-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; border-bottom: 1px solid var(--line); padding-bottom: 10px; margin-bottom: 14px; }
    h2 { margin: 0; font: 500 28px/1.2 Georgia, serif; letter-spacing: 0; }
    .result-meta { color: var(--muted); font-size: 12px; }
    pre { max-width: 100%; min-height: 120px; margin: 0; padding: 20px; overflow: auto; border: 1px solid var(--line); background: #fff; color: #1b302e; font: 13px/1.6 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
    [hidden] { display: none !important; }
    @media (max-width: 620px) {
      .topline, main { width: min(100% - 28px, 960px); }
      main { padding-top: 36px; }
      .form-footer { align-items: flex-start; flex-direction: column; }
      .result-head { align-items: flex-start; flex-direction: column; }
      .model { max-width: 52%; }
    }
  </style>
</head>
<body>
  <header><div class="topline"><a class="brand" href="/">Code / CrewAI</a><span class="model">Hugging Face / Llama 3 8B Instruct</span></div></header>
  <main>
    <h1>Turn a coding task into a working draft.</h1>
    <p class="intro">Describe the behavior you need. The coder agent returns code and example checks.</p>
    <form id="coding-form">
      <label for="assignment">Coding task</label>
      <textarea id="assignment" name="assignment" minlength="10" maxlength="2000" placeholder="Create a Python function that checks whether a string is a palindrome, with example inputs and tests." required></textarea>
      <div class="form-footer"><span id="status" role="status" aria-live="polite"></span><button id="submit" type="submit">Generate code</button></div>
    </form>
    <p class="notice">Generated code is not executed by this public service. Review it and run it in your own sandbox before use.</p>
    <section id="result" aria-live="polite" hidden>
      <div class="result-head"><div><h2>Generated result</h2><span id="result-meta" class="result-meta"></span></div><a id="download" class="download" href="#" download="coding-result.md">Download result</a></div>
      <pre id="output"></pre>
    </section>
  </main>
  <script>
    const form = document.querySelector('#coding-form');
    const button = document.querySelector('#submit');
    const status = document.querySelector('#status');
    const result = document.querySelector('#result');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      status.classList.remove('error');
      status.textContent = 'Starting the coding agent...';
      result.hidden = true;
      try {
        const response = await fetch('/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ assignment: document.querySelector('#assignment').value })
        });
        const started = await response.json();
        if (!response.ok) throw new Error(started.detail || 'The task could not be started.');
        let data;
        while (true) {
          await new Promise((resolve) => setTimeout(resolve, 2000));
          const poll = await fetch(`/jobs/${started.job_id}`);
          data = await poll.json();
          if (!poll.ok) throw new Error(data.detail || 'The task status could not be read.');
          if (data.status === 'completed') break;
          if (data.status === 'failed') throw new Error(data.detail);
          status.textContent = data.status === 'queued' ? 'Waiting for the coder...' : 'The coder is drafting your solution...';
        }
        document.querySelector('#output').textContent = data.result;
        document.querySelector('#result-meta').textContent = 'Execution not performed';
        document.querySelector('#download').href = URL.createObjectURL(new Blob([data.result], { type: 'text/markdown' }));
        result.hidden = false;
        status.textContent = 'Code draft ready';
      } catch (error) {
        status.textContent = error.message;
        status.classList.add('error');
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _run_job(job_id: str, assignment: str) -> None:
    global active_job
    with jobs_lock:
        jobs[job_id]["status"] = "running"

    safe_assignment = (
        f"{assignment}\n\n"
        "Do not execute generated code or claim it was executed. Include the code, "
        "example inputs, and test code in your response. Label any predicted output "
        "as illustrative rather than executed."
    )
    try:
        crew = Coder(allow_code_execution=False).crew()
        for task in crew.tasks:
            task.output_file = None
        result = crew.kickoff(inputs={"assignment": safe_assignment})
        output = result.raw or "\n\n".join(task.raw for task in result.tasks_output)
        if not output.strip():
            raise RuntimeError("The coding agent returned an empty result.")
        update = {"status": "completed", "result": output}
    except Exception:
        logger.exception("Coding task failed")
        update = {
            "status": "failed",
            "detail": "Coding task failed. Check the configured model and server logs.",
        }

    with jobs_lock:
        jobs[job_id].update(update)
        active_job = None


@app.post("/jobs", status_code=202)
def create_job(request: CodingRequest) -> dict[str, str]:
    global active_job
    assignment = request.assignment.strip()
    if len(assignment) < 10:
        raise HTTPException(status_code=422, detail="Describe a coding task using at least 10 characters.")

    with jobs_lock:
        if active_job is not None:
            raise HTTPException(status_code=409, detail="A coding task is already running. Try again shortly.")
        finished_jobs = [job_id for job_id, job in jobs.items() if job["status"] in {"completed", "failed"}]
        while len(jobs) >= 25 and finished_jobs:
            del jobs[finished_jobs.pop(0)]
        if len(jobs) >= 25:
            raise HTTPException(status_code=503, detail="The result store is full. Try again later.")

        job_id = str(uuid4())
        jobs[job_id] = {"status": "queued"}
        active_job = job_id

    try:
        executor.submit(_run_job, job_id, assignment)
    except Exception as error:
        with jobs_lock:
            jobs.pop(job_id, None)
            active_job = None
        raise HTTPException(status_code=503, detail="The coding task could not be queued.") from error
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, str]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Coding task not found.")
        return job.copy()