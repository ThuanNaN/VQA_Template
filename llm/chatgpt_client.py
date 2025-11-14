from __future__ import annotations

import os
from typing import Iterable, Dict, Any, Iterator
import openai
from .base import LLMClient
from dotenv import load_dotenv
load_dotenv()


class ChatGPTClient(LLMClient):
    """Simple ChatGPT/OpenAI Chat client wrapper.

    Reads `OPENAI_API_KEY` from environment. Optional `OPENAI_API_BASE` can change endpoint.
    """

    def __init__(self, *, default_model: str = "gpt-4o", api_key: str | None = None):
        super().__init__(default_model=default_model)
        if openai is None:
            raise RuntimeError("`openai` package is required for ChatGPTClient. Install openai.")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")

        openai.api_key = self.api_key
        base = os.getenv("OPENAI_API_BASE")
        if base:
            openai.api_base = base

    def chat(self, messages: Iterable[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        model = kwargs.pop("model", self.default_model)
        # Convert messages to list
        msgs = list(messages)
        resp = openai.ChatCompletion.create(model=model, messages=msgs, **kwargs)
        return resp

    def stream_chat(self, messages: Iterable[Dict[str, Any]], **kwargs) -> Iterator[str]:
        model = kwargs.pop("model", self.default_model)
        msgs = list(messages)
        stream = openai.ChatCompletion.create(model=model, messages=msgs, stream=True, **kwargs)
        # stream yields events with 'choices'
        try:
            for chunk in stream:
                # chunk may contain 'choices' with 'delta' keys
                choices = chunk.get("choices") or []
                for c in choices:
                    delta = c.get("delta", {})
                    text = delta.get("content") or delta.get("message", {}).get("content")
                    if text:
                        yield text
        except Exception:
            # Fallback: try to return full response
            full = self.chat(msgs, model=model, **kwargs)
            content = ""
            try:
                content = full["choices"][0]["message"]["content"]
            except Exception:
                content = str(full)
            yield content


def example():
    """Example usage (requires OPENAI_API_KEY)."""
    client = ChatGPTClient(default_model="gpt-4o")
    msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Say hello."}]
    resp = client.chat(msgs)
    print(resp)


if __name__ == "__main__":
    example()
