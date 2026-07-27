from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from dataspace_baselines.relays.anthropic_messages import (
    OMITTED_IMAGE_TEXT,
    _RelayServer,
    _normalize_request_bytes,
    _upstream_url,
    normalize_anthropic_payload,
)


def _image(label: str) -> dict[str, object]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": label,
        },
    }


class _UpstreamHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.server.payload = json.loads(  # type: ignore[attr-defined]
            self.rfile.read(length)
        )
        response = b'{"type":"message","content":[]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


class AnthropicMessagesRelayTests(unittest.TestCase):
    def test_retains_only_the_five_most_recent_images(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [_image(str(index)) for index in range(7)],
                }
            ]
        }

        normalized, stats = normalize_anthropic_payload(
            payload, max_images=5
        )
        content = normalized["messages"][0]["content"]

        self.assertEqual(
            stats,
            {
                "images_before": 7,
                "images_after": 5,
                "images_omitted": 2,
            },
        )
        self.assertEqual(
            content[:2],
            [
                {"type": "text", "text": OMITTED_IMAGE_TEXT},
                {"type": "text", "text": OMITTED_IMAGE_TEXT},
            ],
        )
        self.assertEqual(
            [block["source"]["data"] for block in content[2:]],
            ["2", "3", "4", "5", "6"],
        )

    def test_preserves_nested_tool_result_structure(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": [_image("old"), {"type": "text", "text": "ok"}],
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [_image(str(index)) for index in range(5)],
                },
            ]
        }

        normalized, stats = normalize_anthropic_payload(
            payload, max_images=5
        )
        tool_result = normalized["messages"][0]["content"][0]

        self.assertEqual(stats["images_omitted"], 1)
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(tool_result["tool_use_id"], "tool-1")
        self.assertEqual(
            tool_result["content"][0],
            {"type": "text", "text": OMITTED_IMAGE_TEXT},
        )
        self.assertEqual(tool_result["content"][1]["text"], "ok")

    def test_leaves_requests_with_at_most_five_images_unchanged(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": [_image("one"), _image("two")]}
            ]
        }
        before = json.dumps(payload, sort_keys=True)

        normalized, stats = normalize_anthropic_payload(
            payload, max_images=5
        )

        self.assertEqual(json.dumps(normalized, sort_keys=True), before)
        self.assertEqual(stats["images_omitted"], 0)

    def test_normalizes_serialized_request_body(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [_image(str(index)) for index in range(6)],
                }
            ]
        }

        normalized, stats = _normalize_request_bytes(
            json.dumps(payload).encode("utf-8"), max_images=5
        )
        value = json.loads(normalized)

        self.assertEqual(stats["images_omitted"], 1)
        self.assertEqual(
            sum(
                block.get("type") == "image"
                for block in value["messages"][0]["content"]
            ),
            5,
        )

    def test_preserves_upstream_messages_path(self) -> None:
        self.assertEqual(
            _upstream_url(
                "https://ai-gateway.vercel.sh",
                "/v1/messages?beta=true",
            ),
            "https://ai-gateway.vercel.sh/v1/messages?beta=true",
        )

    def test_http_relay_normalizes_before_forwarding(self) -> None:
        try:
            upstream = ThreadingHTTPServer(
                ("127.0.0.1", 0), _UpstreamHandler
            )
        except PermissionError:
            self.skipTest("loopback sockets are unavailable in this sandbox")
        upstream_thread = threading.Thread(
            target=upstream.serve_forever, daemon=True
        )
        upstream_thread.start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        with tempfile.TemporaryDirectory() as temporary:
            event_log = Path(temporary) / "relay.jsonl"
            upstream_base = (
                f"http://127.0.0.1:{upstream.server_address[1]}"
            )
            relay = _RelayServer(
                upstream_base, max_images=5, event_log=event_log
            )
            relay_thread = threading.Thread(
                target=relay.serve_forever, daemon=True
            )
            relay_thread.start()
            try:
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                _image(str(index)) for index in range(6)
                            ],
                        }
                    ]
                }
                request = Request(
                    "http://127.0.0.1:"
                    f"{relay.server_address[1]}/v1/messages",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(
                        json.loads(response.read())["type"], "message"
                    )
            finally:
                relay.shutdown()
                relay.server_close()
                relay_thread.join(timeout=5)

            forwarded = upstream.payload  # type: ignore[attr-defined]
            content = forwarded["messages"][0]["content"]
            self.assertEqual(
                sum(block["type"] == "image" for block in content), 5
            )
            event = json.loads(event_log.read_text(encoding="utf-8"))
            self.assertEqual(event["images_before"], 6)
            self.assertEqual(event["images_after"], 5)


if __name__ == "__main__":
    unittest.main()
