from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


OMITTED_IMAGE_TEXT = (
    "[Earlier image omitted because the model backend accepts at most "
    "five images per request.]"
)


def _is_image_block(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") == "image"


def _collect_image_blocks(value: Any, target: list[dict[str, Any]]) -> None:
    if _is_image_block(value):
        target.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_image_blocks(item, target)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_image_blocks(item, target)


def _replace_omitted_images(value: Any, omitted: set[int]) -> Any:
    if _is_image_block(value):
        if id(value) in omitted:
            return {"type": "text", "text": OMITTED_IMAGE_TEXT}
        return value
    if isinstance(value, list):
        return [_replace_omitted_images(item, omitted) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_omitted_images(item, omitted)
            for key, item in value.items()
        }
    return value


def normalize_anthropic_payload(
    value: Any, *, max_images: int
) -> tuple[Any, dict[str, int]]:
    """Retain only the most recent image blocks in an Anthropic request.

    Claude Code can accumulate image-bearing tool results across turns. Some
    compatible backends accept fewer images than the stock CLI keeps in its
    message history. Replacing older image blocks with text preserves message
    and tool-result structure while enforcing the backend request limit.
    """

    if max_images <= 0:
        raise ValueError("max_images must be positive")
    if not isinstance(value, dict):
        return value, {"images_before": 0, "images_after": 0, "images_omitted": 0}
    messages = value.get("messages")
    if not isinstance(messages, list):
        return value, {"images_before": 0, "images_after": 0, "images_omitted": 0}

    images: list[dict[str, Any]] = []
    _collect_image_blocks(messages, images)
    omitted_count = max(0, len(images) - max_images)
    if omitted_count:
        omitted = {id(block) for block in images[:omitted_count]}
        value["messages"] = _replace_omitted_images(messages, omitted)
    return value, {
        "images_before": len(images),
        "images_after": len(images) - omitted_count,
        "images_omitted": omitted_count,
    }


def _normalize_request_bytes(
    payload: bytes, *, max_images: int
) -> tuple[bytes, dict[str, int]]:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return payload, {
            "images_before": 0,
            "images_after": 0,
            "images_omitted": 0,
        }
    normalized, stats = normalize_anthropic_payload(
        value, max_images=max_images
    )
    return (
        json.dumps(
            normalized, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8"),
        stats,
    )


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

    def __init__(
        self,
        upstream_base: str,
        *,
        max_images: int,
        event_log: Path | None,
    ):
        super().__init__(("127.0.0.1", 0), _RelayHandler)
        self.upstream_base = upstream_base
        self.max_images = max_images
        self.event_log = event_log
        self.event_log_lock = threading.Lock()

    def record_normalization(self, path: str, stats: dict[str, int]) -> None:
        if stats["images_omitted"] <= 0 or self.event_log is None:
            return
        event = {
            "event": "image_history_normalized",
            "request_path": urlsplit(path).path,
            **stats,
        }
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log_lock:
            with self.event_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                )


class _RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _forward(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(content_length)
        server = self.server
        if not isinstance(server, _RelayServer):
            self._send_gateway_error()
            return
        request_body, stats = _normalize_request_bytes(
            request_body, max_images=server.max_images
        )
        server.record_normalization(self.path, stats)

        excluded_request_headers = {
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
            if key.lower() not in excluded_request_headers
        }
        headers["Accept-Encoding"] = "identity"
        upstream_url = _upstream_url(server.upstream_base, self.path)
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
        excluded_response_headers = {
            "connection",
            "content-encoding",
            "content-length",
            "date",
            "proxy-connection",
            "server",
            "transfer-encoding",
        }
        for key, value in response.headers.items():
            if key.lower() not in excluded_response_headers:
                self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        content_type = response.headers.get("Content-Type", "")
        try:
            if "text/event-stream" in content_type.lower():
                for line in response:
                    self.wfile.write(line)
                    self.wfile.flush()
            else:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-base", required=True)
    parser.add_argument("--max-images", required=True, type=int)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.max_images <= 0:
        parser.error("--max-images must be positive")
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a Claude Code command is required after --")

    server = _RelayServer(
        args.upstream_base,
        max_images=args.max_images,
        event_log=args.event_log,
    )
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    child_environment = os.environ.copy()
    child_environment["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    try:
        return subprocess.run(
            command, check=False, env=child_environment
        ).returncode
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
