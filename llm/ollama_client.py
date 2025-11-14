from __future__ import annotations

import os
from typing import Iterable, Dict, Any, Iterator
import ollama
from .base import LLMClient
from dotenv import load_dotenv
load_dotenv()

class OllamaClient(LLMClient):
    """Ollama local LLM client wrapper.

    Connects to a local Ollama server (default: http://localhost:11434).
    Set OLLAMA_HOST environment variable to override.
    """

    def __init__(
        self, 
        *, 
        default_model: str = "llama3",
        host: str | None = None
    ):
        super().__init__(default_model=default_model)
        if ollama is None:
            raise RuntimeError("`ollama` package is required for OllamaClient. Install ollama.")

        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=self.host)

    def list_local_models(self) -> list[str]:
        """List all models available locally in Ollama."""
        try:
            models = self.client.list()
            return [model['model'] for model in models.get('models', [])]
        except Exception as e:
            print(f"[ERROR] Failed to list Ollama models: {e}")
            return []

    def chat(self, messages: Iterable[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Send chat messages to Ollama and return the full response.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional parameters (model, temperature, etc.)
        
        Returns:
            Full response dict from Ollama API
        """
        model = kwargs.pop("model", self.default_model)
        msgs = list(messages)
        
        try:
            response = self.client.chat(
                model=model,
                messages=msgs,
                stream=False,
                **kwargs
            )
            return response
        except ollama.ResponseError as e:
            raise RuntimeError(f"Ollama chat error: {e.error}") from e

    def stream_chat(self, messages: Iterable[Dict[str, Any]], **kwargs) -> Iterator[str]:
        """Stream chat response from Ollama.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional parameters (model, temperature, etc.)
        
        Yields:
            Text chunks from the streaming response
        """
        model = kwargs.pop("model", self.default_model)
        msgs = list(messages)
        
        try:
            stream = self.client.chat(
                model=model,
                messages=msgs,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                # Extract content from the message
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except ollama.ResponseError as e:
            raise RuntimeError(f"Ollama stream error: {e.error}") from e

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion for a single prompt (non-chat mode).
        
        Args:
            prompt: The prompt string
            **kwargs: Additional parameters (model, temperature, etc.)
        
        Returns:
            Full response dict from Ollama API
        """
        model = kwargs.pop("model", self.default_model)
        
        try:
            response = self.client.generate(
                model=model,
                prompt=prompt,
                stream=False,
                **kwargs
            )
            return response
        except ollama.ResponseError as e:
            raise RuntimeError(f"Ollama generate error: {e.error}") from e

    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream generation for a single prompt.
        
        Args:
            prompt: The prompt string
            **kwargs: Additional parameters (model, temperature, etc.)
        
        Yields:
            Text chunks from the streaming response
        """
        model = kwargs.pop("model", self.default_model)
        
        try:
            stream = self.client.generate(
                model=model,
                prompt=prompt,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']
                    
        except ollama.ResponseError as e:
            raise RuntimeError(f"Ollama stream generate error: {e.error}") from e


def example_chat():
    """Example usage with chat interface."""
    client = OllamaClient(default_model="llama3")
    
    # List available models
    print("Available models:", client.list_local_models())
    
    # Chat example
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the difference between a VM and a container in one paragraph."}
    ]
    
    print("\n--- Non-streaming chat ---")
    response = client.chat(messages)
    print(response['message']['content'])
    
    print("\n--- Streaming chat ---")
    for chunk in client.stream_chat(messages):
        print(chunk, end='', flush=True)
    print("\n")


def example_generate():
    """Example usage with generate interface (non-chat)."""
    client = OllamaClient(default_model="llama3")
    
    prompt = "What is the capital of France?"
    
    print("\n--- Non-streaming generation ---")
    response = client.generate(prompt)
    print(response['response'])
    
    print("\n--- Streaming generation ---")
    for chunk in client.stream_generate(prompt):
        print(chunk, end='', flush=True)
    print("\n")


if __name__ == "__main__":
    print("=== Chat Examples ===")
    example_chat()
    
    print("\n=== Generate Examples ===")
    example_generate()