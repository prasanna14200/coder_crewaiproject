import os
import sys

from crewai import LLM


MODEL = "gemini/gemini-3.8-flash"


def request_model(api_key: str | None = None) -> str:
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    response = LLM(
        model=MODEL,
        api_key=key,
        timeout=20,
        max_tokens=48,
        num_retries=0,
    ).call("Reply with exactly: direct Gemini request succeeded")
    return str(response).strip()


def main() -> int:
    try:
        response = request_model()
    except Exception as error:
        if isinstance(error, RuntimeError) and "GEMINI_API_KEY" in str(error):
            print("GEMINI_API_KEY is not configured. Set it before running the direct Gemini check.")
            return 1
        response_object = getattr(error, "response", None)
        status_code = getattr(error, "status_code", None) or getattr(response_object, "status_code", None)
        error_code = str(getattr(error, "code", "")).upper()
        error_name = type(error).__name__
        if status_code == 429 or error_code == "RESOURCE_EXHAUSTED":
            print("Direct Gemini request hit the Free-tier quota. Check the active limits in Google AI Studio.")
        elif status_code in {401, 403} or error_code == "API_KEY_INVALID":
            print("Direct Gemini request was rejected. Check GEMINI_API_KEY and its Google AI Studio project.")
        else:
            print(f"Direct Gemini request failed ({error_name}, HTTP {status_code or 'unknown'}). Check model access and server logs.")
        return 1

    print("Direct Gemini request succeeded.")
    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
