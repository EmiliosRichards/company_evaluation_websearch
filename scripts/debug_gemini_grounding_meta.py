import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


def main() -> int:
    load_dotenv(override=False)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY")

    model = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
    client = genai.Client(api_key=api_key)

    prompt = (
        "Use Google Search grounding at least once. Then answer with exactly: OK.\n"
        "This is just a debugging request to inspect grounding metadata."
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
        ),
    )

    um = getattr(resp, "usage_metadata", None)
    in_tok = getattr(um, "prompt_token_count", None)
    out_tok = getattr(um, "candidates_token_count", None)
    print(f"model={model}")
    print(f"usage.prompt_token_count={in_tok} usage.candidates_token_count={out_tok}")

    candidates = getattr(resp, "candidates", None) or []
    print(f"candidates={len(candidates)}")
    if not candidates:
        return 0

    cand0 = candidates[0]
    gm = getattr(cand0, "grounding_metadata", None)
    print(f"has_grounding_metadata={gm is not None}")
    if gm is None:
        return 0

    # Print known/likely fields and their lengths if present.
    for name in [
        "web_search_queries",
        "search_queries",
        "grounding_chunks",
        "grounding_attributions",
        "retrieval_metadata",
        "search_entry_point",
        "grounding_supports",
    ]:
        if hasattr(gm, name):
            v = getattr(gm, name)
            try:
                ln = len(v)
            except Exception:
                ln = None
            print(f"grounding_metadata.{name}: present len={ln}")

    fields = [a for a in dir(gm) if not a.startswith("_")]
    print("grounding_metadata fields (first 120):")
    for f in fields[:120]:
        print(f"- {f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


