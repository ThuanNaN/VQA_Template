from __future__ import annotations

import os
from typing import Iterable, Dict, Any, Iterator
from google import genai
from .base import LLMClient
from dotenv import load_dotenv
load_dotenv()

class GeminiClient(LLMClient):
    """Wrapper for Google Generative AI / Gemini chat API.

    Uses `GOOGLE_API_KEY` env var by default or application default credentials.
    """

    def __init__(self, *, default_model: str = "gemini-1.5", api_key: str | None = None):
        super().__init__(default_model=default_model)
        if genai is None:
            raise RuntimeError("`google.generativeai` package is required for GeminiClient. Install google-generative-ai")

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def chat(self, messages: Iterable[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        model = kwargs.pop("model", self.default_model)
        msgs = list(messages)
        # Convert to Gemini format
        resp = genai.chat.create(model=model, messages=msgs, **kwargs)
        return resp

    def stream_chat(self, messages: Iterable[Dict[str, Any]], **kwargs) -> Iterator[str]:
        # The Python client supports streaming via `stream` context manager
        model = kwargs.pop("model", self.default_model)
        msgs = list(messages)
        try:
            for event in genai.chat.stream(model=model, messages=msgs, **kwargs):
                # event might be string chunks or objects with 'delta' / 'content'
                text = None
                if isinstance(event, str):
                    text = event
                else:
                    # safe extraction
                    text = getattr(event, "content", None) or (event.get("delta") if isinstance(event, dict) else None)
                if text:
                    yield text
        except Exception:
            # fallback: return non-streamed result
            full = self.chat(msgs, model=model, **kwargs)
            try:
                yield full["candidates"][0]["content"]
            except Exception:
                yield str(full)


def example():
    client = GeminiClient(default_model="gemini-1.5")
    msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hi from VQA template."}]
    resp = client.chat(msgs)
    print(resp)


if __name__ == "__main__":
    example()
