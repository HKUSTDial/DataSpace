from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataspace_baselines.relays.openai_chat import (
    RELAY_BASE_PLACEHOLDER,
    _configure_local_base,
    _normalize_sse_line,
    _upstream_url,
    normalize_chat_payload,
)


class OpenAIChatRelayTests(unittest.TestCase):
    def test_normalizes_only_unknown_finish_reason(self) -> None:
        payload = {
            "choices": [
                {"finish_reason": "other", "delta": {"content": "done"}},
                {"finish_reason": "tool_calls"},
                {"finish_reason": None},
            ]
        }
        self.assertIs(normalize_chat_payload(payload), payload)
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(
            payload["choices"][0]["delta"]["content"], "done"
        )
        self.assertEqual(payload["choices"][1]["finish_reason"], "tool_calls")
        self.assertIsNone(payload["choices"][2]["finish_reason"])

    def test_normalizes_stream_event_without_changing_content(self) -> None:
        line = (
            b'data: {"choices":[{"delta":{"content":"answer"},'
            b'"finish_reason":"other"}]}\n'
        )
        normalized = _normalize_sse_line(line)
        value = json.loads(normalized.removeprefix(b"data: "))
        self.assertEqual(value["choices"][0]["delta"]["content"], "answer")
        self.assertEqual(value["choices"][0]["finish_reason"], "stop")

    def test_preserves_upstream_v1_prefix(self) -> None:
        self.assertEqual(
            _upstream_url(
                "https://ai-gateway.vercel.sh/v1",
                "/v1/chat/completions?mode=test",
            ),
            "https://ai-gateway.vercel.sh/v1/chat/completions?mode=test",
        )

    def test_launcher_replaces_one_task_local_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text(
                f'base_url = "{RELAY_BASE_PLACEHOLDER}"\n', encoding="utf-8"
            )
            _configure_local_base(path, 43210)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                'base_url = "http://127.0.0.1:43210/v1"\n',
            )


if __name__ == "__main__":
    unittest.main()
