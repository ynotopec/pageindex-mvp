import json
from typing import Generator, List

import requests


def _openai_chat_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    if clean.endswith("/v1"):
        return f"{clean}/chat/completions"
    return f"{clean}/v1/chat/completions"


def ollama_chat_stream(model: str, host: str, messages: List[dict]) -> Generator[str, None, None]:
    url = f"{host.rstrip('/')}/api/chat"
    payload = {"model": model, "stream": True, "messages": messages}
    with requests.post(url, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = requests.models.complexjson.loads(line)  # type: ignore[attr-defined]
            except Exception:
                data = json.loads(line)

            if "message" in data and data["message"] and "content" in data["message"]:
                chunk = data["message"]["content"]
                if chunk:
                    yield chunk

            if data.get("done") is True:
                break


def openai_compatible_chat_stream(
    model: str,
    base_url: str,
    api_key: str,
    messages: List[dict],
) -> Generator[str, None, None]:
    if not api_key.strip():
        raise ValueError("OPENAI API key manquante")

    url = _openai_chat_url(base_url)
    payload = {"model": model, "stream": True, "messages": messages}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with requests.post(url, json=payload, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data_part = line[5:].strip()
            if data_part == "[DONE]":
                break
            try:
                data = requests.models.complexjson.loads(data_part)  # type: ignore[attr-defined]
            except Exception:
                data = json.loads(data_part)
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            chunk = delta.get("content")
            if chunk:
                yield chunk
