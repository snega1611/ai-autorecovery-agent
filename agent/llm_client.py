from __future__ import annotations
import json
import os
import requests
from typing import List, Dict, Any, Optional

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("RECALL_MODEL", "qwen3:1.7b")


class LLMError(RuntimeError):
    pass


def chat(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Sends a chat request to Ollama. Returns the raw message dict from the
    model, which may contain a `tool_calls` field if the model chose to call
    a tool.

    temperature is kept low (0.1) deliberately -- this agent should be
    consistent and literal about what the transcript says, not creative.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise LLMError(
            f"Could not reach Ollama at {OLLAMA_URL}. Is `ollama serve` running "
            f"and have you run `ollama pull {MODEL_NAME}`? Original error: {e}"
        )

    data = resp.json()
    message = data.get("message", {})
    if not message:
        raise LLMError(f"Unexpected Ollama response shape: {data}")
    return message


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Fallback parser: pulls the first {...} JSON object out of free text,
    for models/prompts that reply in JSON-instructed prose rather than
    proper tool_calls."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    # manual smoke test -- requires `ollama serve` running locally
    msg = chat([{"role": "user", "content": "Reply with exactly the word: pong"}])
    print(msg)
