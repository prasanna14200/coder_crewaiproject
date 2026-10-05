# CrewAI Coder

This project uses one CrewAI Python developer agent to generate code for a user-supplied assignment. The sequential crew runs the configured `coding_task`; the CLI writes its result to `output/code_and_output.txt`, while the web app returns it as a downloadable response without writing per-request files.

## Model and configuration

The agent uses `huggingface/meta-llama/Meta-Llama-3-8B-Instruct` as configured in `src/coder/config/agents.yaml`. Set `HUGGINGFACE_API_KEY` to a Hugging Face access token that can use Inference Providers. `MODEL` optionally overrides the model identifier; the application does not switch providers automatically. The custom example tool is not attached to this crew, so `SERPER_API_KEY` is not required. Crew tracing is optional and disabled by default.

Copy `.env.example` to `.env` for local development. Keep credentials out of Git. If Hugging Face responds with `401 Unauthorized`, replace the token with a valid Inference Providers token. If it reports `model_not_supported`, enable an inference provider for the selected model or select a Hugging Face model available to your token.

## Install and run

Use Python 3.10 through 3.13 and install the locked dependencies in the project virtual environment:

```bash
uv sync --locked
```

The existing CLI uses CrewAI safe code execution and requires a running Docker daemon. Give it a coding assignment as an argument, or omit the argument to run the original series assignment:

```bash
uv run coder "Create a Python palindrome function with example inputs and tests."
```

The hosted web application deliberately disables code execution. It shows and downloads the generated result, but does not run it or claim that its example output has been tested. Review generated code and run it in a sandbox you control.

Run the web app locally:

```bash
uv run python -m uvicorn coder.web:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. The app accepts one coding job at a time and polls for the result. Its health endpoint is `/healthz`. Job state is held in memory and is lost when the process restarts.

Run the web-interface tests:

```bash
uv run --no-sync python -m unittest discover -s tests -v
```

## Render deployment

Push the project changes to `main`, then create a **Web Service** from `https://github.com/prasanna14200/coder_crewaiproject` on branch `main`.

| Setting | Value |
| --- | --- |
| Root Directory | Leave blank |
| Runtime | Python 3.11 |
| Build Command | `pip install uv && uv sync --frozen --no-dev` |
| Start Command | `uv run --no-sync python -m uvicorn coder.web:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/healthz` |

Set `PYTHON_VERSION=3.11.11` and `HUGGINGFACE_API_KEY` in the Render Environment settings. Get the token from Hugging Face **Settings > Access Tokens** and grant it Inference Providers permissions. Set `MODEL` only if overriding the configured Hugging Face model. Render supplies `PORT`; do not set it yourself. Do not add the key to the repository or build command.

This service calls Hugging Face remotely; it does not load local model weights. The web route disables code execution, so Render does not need Docker. The CrewAI dependency set is substantial and model calls can take time. Use a single paid `1c-2g` instance (1 CPU, 2 GB RAM) for reliable use. Render Free provides 512 MB RAM and 0.1 CPU and spins down after 15 minutes idle; treat it only as a best-effort smoke test. The app serializes jobs in one process; do not scale it to multiple instances without adding shared job storage/queueing. A restart clears active and completed in-memory jobs.

After deployment, verify `/healthz` returns `{"status":"ok"}`, submit a coding task, wait for the generated result, and test the download link. A successful health check does not verify Hugging Face access; a completed coding task does.
