#!/usr/bin/env python3
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Interactive CLI to chat with the API server.

Examples:
  BASE_URL=http://localhost:8000 MODEL_NAME=/path/to/model python3 chat_cli.py
  python3 chat_cli.py --base-url http://localhost:8000 --model my-model --stream

Commands during chat:
  /exit    Exit the program
  /quit    Same as /exit
  /reset   Reset the conversation history
  /system <text>  Set/replace system prompt
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import requests


def _normalize_base_url(url: str) -> str:
    return url[:-1] if url.endswith('/') else url


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, color_code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{color_code}m{text}\033[0m"


def _post_chat(
    base_url: str,
    payload: Dict,
    timeout_sec: float,
) -> requests.Response:
    return requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout_sec,
        stream=bool(payload.get("stream", False)),
    )


def chat_once(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_sec: float,
) -> tuple[str, str, dict]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        # keep temperature/top_p for servers that honor them; harmless otherwise
        "temperature": temperature,
        "top_p": top_p,
    }
    start_time = time.time()
    resp = _post_chat(base_url, payload, timeout_sec)
    latency = time.time() - start_time
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    # Match parsing in test_api.py
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage")
    meta = f"  {latency:.2f}s"
    if usage:
        meta += f"  tokens: {usage}"
    return content, meta, data


def chat_stream(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    block_length: int,
    timeout_sec: float,
) -> str:
    # Match the structure used in test_api for streaming
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "block_length": block_length,
        "stream": True,
    }
    resp = _post_chat(base_url, payload, timeout_sec)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    full_text_parts: List[str] = []
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="ignore")
        # Accept both 'data:' and 'data: '
        if not (line.startswith("data:") or line.startswith("data: ")):
            continue
        data_str = line[5:]
        if data_str.startswith(" "):
            data_str = data_str[1:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if "choices" not in chunk or not chunk["choices"]:
            continue
        delta = chunk["choices"][0].get("delta", {})
        content_piece = delta.get("content", "")
        if content_piece:
            full_text_parts.append(content_piece)
            # Stream to stdout without newline
            print(content_piece, end="", flush=True)
    print()  # newline after stream finishes
    if not full_text_parts:
        print(_c("(no tokens received)", "90"))
    return "".join(full_text_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive API chat client")
    parser.add_argument(
        "--base-url",
        default=_normalize_base_url(os.environ.get("BASE_URL", "http://localhost:8000")),
        help="Base URL of the API, e.g., http://localhost:8000",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "MODEL_NAME",
            "Salesforce/CoDA-v0-Instruct",
        ),
        help="Model name/path to use",
    )
    parser.add_argument(
        "--system",
        default=os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant."),
        help="Initial system prompt",
    )
    parser.add_argument("--stream", action="store_true", help="Use streaming responses")
    parser.add_argument("--show-meta", action="store_true", help="Show latency and token usage")
    parser.add_argument("--debug", action="store_true", help="Print raw JSON when content is empty")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling")
    parser.add_argument("--block-length", type=int, default=512, help="Streaming block length")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout seconds")

    args = parser.parse_args()
    base_url = _normalize_base_url(args.base_url)
    model = args.model or os.environ.get(
        "MODEL_NAME",
        "Salesforce/CoDA-v0-Instruct",
    )

    system_prompt = args.system
    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": system_prompt})

    print(_c("API Chat", "96"))
    print("Base URL:", base_url)
    print("Model:", model)
    print(_c("Type your message and press Enter. Use /exit to quit.", "90"))

    try:
        while True:
            try:
                user_input = input(_c("You: ", "92"))
            except EOFError:
                print()
                break

            if not user_input.strip():
                continue

            # Commands
            if user_input.strip().lower() in {"/exit", "/quit"}:
                break
            if user_input.strip().lower() == "/reset":
                messages = [{"role": "system", "content": system_prompt}]
                print(_c("Conversation reset.", "90"))
                continue
            if user_input.strip().startswith("/system "):
                system_prompt = user_input.strip()[8:].strip()
                messages = [{"role": "system", "content": system_prompt}]
                print(_c("System prompt updated.", "90"))
                continue

            messages.append({"role": "user", "content": user_input})

            print(_c("Assistant:", "94"))
            try:
                if args.stream:
                    assistant_text = chat_stream(
                        base_url=base_url,
                        model=model,
                        messages=messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        block_length=args.block_length,
                        timeout_sec=args.timeout,
                    )
                else:
                    assistant_text, meta, raw = chat_once(
                        base_url=base_url,
                        model=model,
                        messages=messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        timeout_sec=args.timeout,
                    )
            except requests.exceptions.ConnectionError as e:
                print(_c(f"Connection error: {e}", "91"))
                # Remove the last user message to allow retry later
                messages.pop()
                continue
            except Exception as e:
                print(_c(f"Error: {e}", "91"))
                messages.pop()
                continue

            # Print assistant text and optional meta
            if assistant_text and assistant_text.strip():
                print(assistant_text)
            else:
                if args.debug:
                    try:
                        print(_c("(debug: empty content, raw JSON follows)", "90"))
                        print(json.dumps(raw, indent=2))
                    except Exception:
                        print(_c("(no content received)", "90"))
                else:
                    print(_c("(no content received)", "90"))
            if args.show_meta:
                print(_c(meta, "90"))

            messages.append({"role": "assistant", "content": assistant_text})

    except KeyboardInterrupt:
        print()
    finally:
        print(_c("Goodbye!", "90"))


if __name__ == "__main__":
    main()


