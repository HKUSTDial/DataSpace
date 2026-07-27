from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


RELAY_BASE_PLACEHOLDER = "http://127.0.0.1:0/v1"


def normalize_chat_payload(value: Any) -> Any:
    """Map non-standard Chat Completions termination labels to ``stop``.

    Vercel can expose the provider-agnostic ``other`` finish reason through
    its OpenAI-compatible endpoint. Grok Build 0.2.106 deserializes the
    OpenAI enum strictly and otherwise aborts an already completed turn.
    """

    if not isinstance(value, dict):
        return value
    choices = value.get("choices")
    if not isinstance(choices, list):
        return value
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        finish_reason = choice.get("finish_reason")
        if finish_reason == "other":
            choice["finish_reason"] = "stop"
    return value


def _normalize_json_bytes(payload: bytes) -> bytes:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return payload
    normalized = normalize_chat_payload(value)
    return json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _normalize_sse_line(line: bytes) -> bytes:
    stripped = line.rstrip(b"\r\n")
    suffix = line[len(stripped) :]
    if not stripped.startswith(b"data:"):
        return line
    payload = stripped[5:].lstrip()
    if not payload or payload == b"[DONE]":
        return line
    return b"data: " + _normalize_json_bytes(payload) + suffix


def _upstream_url(upstream_base: str, incoming_path: str) -> str:
    base = urlsplit(upstream_base)
    incoming = urlsplit(incoming_path)
    base_path = base.path.rstrip("/")
    request_path = incoming.path
    if base_path and request_path.startswith(base_path + "/"):
        request_path = request_path[len(base_path) :]
    path = base_path + "/" + request_path.lstrip("/")
    return urlunsplit((base.scheme, base.netloc, path, incoming.query, ""))


class _RelayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, upstream_base: str):
        super().__init__(("127.0.0.1", 0), _RelayHandler)
        self.upstream_base = upstream_base


class _RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _forward(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(content_length)
        excluded = {
            "accept-encoding",
            "connection",
            "content-length",
            "host",
            "proxy-connection",
            "transfer-encoding",
        }
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in excluded
        }
        headers["Accept-Encoding"] = "identity"
        upstream_url = _upstream_url(
            self.server.upstream_base, self.path  # type: ignore[attr-defined]
        )
        request = Request(
            upstream_url,
            data=request_body,
            headers=headers,
            method=self.command,
        )
        try:
            response = urlopen(request, timeout=900)
        except HTTPError as exc:
            response = exc
        except URLError:
            self._send_gateway_error()
            return

        self.send_response(response.status)
        content_type = response.headers.get(
            "Content-Type", "application/octet-stream"
        )
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            if "text/event-stream" in content_type.lower():
                for line in response:
                    self.wfile.write(_normalize_sse_line(line))
                    self.wfile.flush()
            else:
                self.wfile.write(_normalize_json_bytes(response.read()))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            response.close()

    def _send_gateway_error(self) -> None:
        payload = b'{"error":{"message":"local compatibility relay failed"}}'
        self.send_response(502)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    do_POST = _forward


def _configure_local_base(config_path: Path, port: int) -> None:
    content = config_path.read_text(encoding="utf-8")
    if content.count(RELAY_BASE_PLACEHOLDER) != 1:
        raise RuntimeError("Grok relay base placeholder is missing or ambiguous")
    content = content.replace(
        RELAY_BASE_PLACEHOLDER, f"http://127.0.0.1:{port}/v1"
    )
    config_path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-base", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a Grok Build command is required after --")

    server = _RelayServer(args.upstream_base)
    port = int(server.server_address[1])
    _configure_local_base(args.config, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        return subprocess.run(command, check=False).returncode
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
