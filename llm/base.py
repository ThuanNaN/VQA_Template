from __future__ import annotations

import abc
from typing import Iterable, AsyncIterable, Dict, Any, Optional


class LLMClient(abc.ABC):
    """Abstract base for LLM/chatbot clients.

    Concrete implementations should provide `chat` and `stream_chat` methods.
    """

    def __init__(self, *, default_model: Optional[str] = None):
        self.default_model = default_model

    @abc.abstractmethod
    def chat(self, messages: Iterable[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Send a list of messages and return the model response as a dict.

        messages: sequence of {"role": "user|assistant|system", "content": str}
        """

    @abc.abstractmethod
    def stream_chat(self, messages: Iterable[Dict[str, Any]], **kwargs) -> Iterable[str]:
        """Return an iterator of streamed text chunks for the given messages."""

    async def achat(self, messages: Iterable[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Async wrapper for sync implementations if needed."""
        # Default: call sync method in thread if sync only. Subclasses can override.
        return self.chat(messages, **kwargs)

    async def astream_chat(self, messages: Iterable[Dict[str, Any]], **kwargs) -> AsyncIterable[str]:
        """Async wrapper for streamed responses."""
        for chunk in self.stream_chat(messages, **kwargs):
            yield chunk
