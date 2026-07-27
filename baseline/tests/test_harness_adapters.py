from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dataspace_baselines.config import (
    DataWorkbenchConfig,
    ExperimentModelConfig,
    HarnessConfig,
    HarnessExperimentConfig,
)
from dataspace_baselines.core.task import TaskSpec
from dataspace_baselines.harnesses.claude_code import (
    _MODEL_ENVIRONMENT_KEYS,
    _exit_classification as classify_claude_exit,
    _validate_single_backbone as validate_claude_backbone,
    build_claude_code,
)
from dataspace_baselines.harnesses.claude_code import _trace as claude_trace
from dataspace_baselines.harnesses.codex import build_codex
from dataspace_baselines.harnesses.codex import (
    _exit_classification as classify_codex_exit,
)
from dataspace_baselines.harnesses.codex import _trace as codex_trace
from dataspace_baselines.harnesses.grok_build import (
    _MODEL_ROLES,
    _exit_classification as classify_grok_exit,
    _validate_single_backbone as validate_grok_backbone,
    build_grok_build,
)
from dataspace_baselines.harnesses.grok_build import _trace as grok_trace
from dataspace_baselines.harnesses.native_common import configured_runtime_binds
from dataspace_baselines.harnesses.smolagents import build_smolagents
from dataspace_baselines.harnesses.smolagents import _trace as smolagents_trace
from dataspace_baselines.harnesses.smolagents_entry import (
    _agent_snapshot,
    _container_run_kwargs,
)


def _config(
    harness_key: str, executable: str, root: Path
) -> HarnessExperimentConfig:
    rootfs = root / "workbench-rootfs"
    tools = rootfs / "opt" / "dataspace" / "TOOLS.md"
    tools.parent.mkdir(parents=True, exist_ok=True)
    tools.write_text("test tools\n", encoding="utf-8")
    bwrap = rootfs / "usr" / "bin" / "bwrap"
    bwrap.parent.mkdir(parents=True, exist_ok=True)
    bwrap.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bwrap.chmod(0o755)
    socat = rootfs / "usr" / "bin" / "socat"
    socat.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    socat.chmod(0o755)
    return HarnessExperimentConfig(
        model=ExperimentModelConfig(
            key="mimo-v2.5",
            model="xiaomi/mimo-v2.5",
            api_key_env="TEST_GATEWAY_KEY",
            openai_base_url="https://gateway.test/v1",
            anthropic_base_url="https://gateway.test",
        ),
        wall_time_seconds=1800,
        concurrency=8,
        workbench=DataWorkbenchConfig(
            image="sandbox:test",
            rootfs_path=rootfs,
        ),
        harnesses={
            harness_key: HarnessConfig(
                key=harness_key,
                executable=executable,
            )
        },
    )


def _task(root: Path) -> TaskSpec:
    task_dir = root / "task_1"
    context = task_dir / "context"
    context.mkdir(parents=True)
    (context / "data.csv").write_text("answer\n42\n", encoding="utf-8")
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": "task_1", "question": "Return the answer."}),
        encoding="utf-8",
    )
    return TaskSpec.load(task_dir)


class HarnessAdapterTests(unittest.TestCase):
    def test_runtime_path_expands_pinned_version(self) -> None:
        harness = HarnessConfig(
            key="codex",
            options={
                "version": "1.2.3",
                "runtime_path": "/tmp/codex-{version}",
                "runtime_mount": "/mnt/runtime/codex",
            },
        )
        self.assertEqual(
            configured_runtime_binds(harness),
            ((Path("/tmp/codex-1.2.3"), "/mnt/runtime/codex"),),
        )

    def _invocation(self, builder, key: str, executable: str):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        task = _task(root)
        run_root = root / "run"
        output_root = run_root / "output"
        run_root.mkdir()
        output_root.mkdir()
        config = _config(key, executable, root)
        harness = config.harnesses[key]
        with patch.dict(os.environ, {"TEST_GATEWAY_KEY": "test-secret"}):
            runner = builder(config, harness)
            invocation = runner.invocation_factory(task, run_root, output_root)
        return run_root, invocation

    def test_codex_disables_web_and_command_network(self) -> None:
        run_root, invocation = self._invocation(
            build_codex, "codex", "/bin/codex"
        )
        config = (run_root / "runtime/home/.codex/config.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn('web_search = "disabled"', config)
        self.assertIn("network_access = false", config)
        self.assertIn("supports_websockets = false", config)
        self.assertIn("enable_request_compression = false", config)
        self.assertIn("model_context_window = 200000", config)
        self.assertIn("model_auto_compact_token_limit = 159040", config)
        self.assertIn('exclude = ["TEST_GATEWAY_KEY"', config)
        self.assertIn("allow_login_shell = false", config)
        self.assertIn("--ephemeral", invocation.command)
        self.assertNotIn("test-secret", " ".join(invocation.command))
        self.assertFalse(invocation.inherit_environment)
        self.assertEqual(
            invocation.metadata["request_compression"], "disabled"
        )

    def test_codex_chat_uses_isolated_compatible_runtime(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        task = _task(root)
        run_root = root / "run"
        output_root = run_root / "output"
        runtime = root / "codex-runtime"
        runtime.mkdir()
        run_root.mkdir()
        output_root.mkdir()
        config = _config(
            "codex", "/mnt/runtime/codex-0.80.0/codex/codex", root
        )
        harness = HarnessConfig(
            key="codex",
            executable="/mnt/runtime/codex-0.80.0/codex/codex",
            options={
                "version": "0.80.0",
                "wire_api": "chat",
                "runtime_path": str(runtime),
                "runtime_mount": "/mnt/runtime/codex-0.80.0",
            },
        )
        with patch.dict(os.environ, {"TEST_GATEWAY_KEY": "test-secret"}):
            runner = build_codex(config, harness)
            invocation = runner.invocation_factory(task, run_root, output_root)

        codex_config = (
            run_root / "runtime/home/.codex/config.toml"
        ).read_text(encoding="utf-8")
        self.assertIn('wire_api = "chat"', codex_config)
        self.assertNotIn("supports_websockets", codex_config)
        self.assertNotIn("--strict-config", invocation.command)
        self.assertNotIn("--ephemeral", invocation.command)
        self.assertNotIn("--ignore-rules", invocation.command)
        self.assertIn(str(runtime), invocation.command)
        self.assertIn("/mnt/runtime/codex-0.80.0", invocation.command)
        self.assertEqual(invocation.metadata["wire_api"], "chat_completions")
        self.assertEqual(invocation.metadata["runtime_version"], "0.80.0")

    def test_claude_disables_web_mcp_and_network(self) -> None:
        run_root, invocation = self._invocation(
            build_claude_code, "claude-code", "/bin/claude"
        )
        settings = json.loads(
            (run_root / "runtime/home/.claude/settings.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(settings["permissions"]["deny"], ["WebSearch", "WebFetch"])
        self.assertIn("Bash", settings["permissions"]["allow"])
        self.assertIn("Write", settings["permissions"]["allow"])
        self.assertEqual(settings["sandbox"]["network"]["allowedDomains"], [])
        credential_names = {
            item["name"]
            for item in settings["sandbox"]["credentials"]["envVars"]
        }
        self.assertIn("TEST_GATEWAY_KEY", credential_names)
        self.assertIn("ANTHROPIC_AUTH_TOKEN", credential_names)
        self.assertTrue(settings["sandbox"]["failIfUnavailable"])
        self.assertIn("--strict-mcp-config", invocation.command)
        mcp_index = invocation.command.index("--mcp-config")
        self.assertEqual(
            json.loads(invocation.command[mcp_index + 1]),
            {"mcpServers": {}},
        )
        self.assertIn("--no-session-persistence", invocation.command)
        for key in _MODEL_ENVIRONMENT_KEYS:
            self.assertEqual(
                invocation.environment[key], "xiaomi/mimo-v2.5"
            )
            self.assertEqual(settings["env"][key], "xiaomi/mimo-v2.5")
        self.assertEqual(
            invocation.environment["CLAUDE_CODE_NO_MODEL_FALLBACK"], "1"
        )
        self.assertEqual(
            invocation.environment["CLAUDE_CODE_DISABLE_FAST_MODE"], "1"
        )
        self.assertEqual(
            invocation.environment["CLAUDE_CODE_MAX_CONTEXT_TOKENS"],
            "200000",
        )
        self.assertEqual(
            invocation.environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"],
            "32768",
        )
        self.assertEqual(
            invocation.environment["CLAUDE_CODE_AUTO_COMPACT_WINDOW"],
            "159040",
        )
        self.assertEqual(
            invocation.environment["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"],
            "70",
        )
        self.assertIn(
            "/mnt/runtime/dataspace-relays/anthropic_messages.py",
            invocation.command,
        )
        max_images_index = invocation.command.index("--max-images")
        self.assertEqual(invocation.command[max_images_index + 1], "5")
        self.assertEqual(
            invocation.metadata["image_history_policy"],
            {
                "max_images_per_request": 5,
                "retention": "most_recent",
                "older_images": "text_placeholder",
            },
        )

    def test_claude_requires_nested_sandbox_before_running_tasks(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        config = _config("claude-code", "/bin/claude", root)
        (config.workbench.rootfs_path / "usr/bin/bwrap").unlink()

        with self.assertRaisesRegex(
            RuntimeError,
            "requires bubblewrap and socat inside the Data Workbench rootfs",
        ):
            build_claude_code(config, config.harnesses["claude-code"])

    def test_claude_requires_sandbox_relay_before_running_tasks(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        config = _config("claude-code", "/bin/claude", root)
        (config.workbench.rootfs_path / "usr/bin/socat").unlink()

        with self.assertRaisesRegex(
            RuntimeError,
            "requires bubblewrap and socat inside the Data Workbench rootfs",
        ):
            build_claude_code(config, config.harnesses["claude-code"])

    def test_anthropic_relay_is_claude_only(self) -> None:
        _, claude = self._invocation(
            build_claude_code, "claude-code", "/bin/claude"
        )
        _, codex = self._invocation(build_codex, "codex", "/bin/codex")
        _, grok = self._invocation(
            build_grok_build, "grok-build", "/bin/grok"
        )
        relay = "/mnt/runtime/dataspace-relays/anthropic_messages.py"

        self.assertIn(relay, claude.command)
        self.assertNotIn(relay, codex.command)
        self.assertNotIn(relay, grok.command)

    def test_grok_uses_strict_sandbox_without_web_or_memory(self) -> None:
        run_root, invocation = self._invocation(
            build_grok_build, "grok-build", "/bin/grok"
        )
        config = (run_root / "runtime/home/.grok/config.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn('api_backend = "chat_completions"', config)
        self.assertIn("context_window = 200000", config)
        for role in _MODEL_ROLES:
            self.assertIn(f'{role} = "dataspace-backbone"', config)
        self.assertIn("--disable-web-search", invocation.command)
        self.assertIn("--always-approve", invocation.command)
        self.assertIn("--no-memory", invocation.command)
        sandbox_index = invocation.command.index("--sandbox")
        self.assertEqual(
            invocation.command[sandbox_index + 1], "dataspace-strict"
        )
        sandbox = (run_root / "runtime/home/.grok/sandbox.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn("extends = \"strict\"", sandbox)
        self.assertIn('read_only = ["/opt/dataspace"]', sandbox)

    def test_grok_finish_reason_relay_is_transport_only(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        task = _task(root)
        run_root = root / "run"
        output_root = run_root / "output"
        run_root.mkdir()
        output_root.mkdir()
        config = _config("grok-build", "/bin/grok", root)
        config.harnesses["grok-build"] = HarnessConfig(
            key="grok-build",
            executable="/bin/grok",
            options={
                "api_backend": "chat_completions",
                "normalize_finish_reasons": True,
            },
        )
        with patch.dict(os.environ, {"TEST_GATEWAY_KEY": "test-secret"}):
            runner = build_grok_build(
                config, config.harnesses["grok-build"]
            )
            invocation = runner.invocation_factory(
                task, run_root, output_root
            )

        content = (run_root / "runtime/home/.grok/config.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn('base_url = "http://127.0.0.1:0/v1"', content)
        self.assertIn(
            "/mnt/runtime/dataspace-relays/openai_chat.py",
            invocation.command,
        )
        self.assertIn("/usr/local/bin/python3", invocation.command)
        self.assertEqual(
            invocation.metadata["finish_reason_normalization"], True
        )
        self.assertEqual(
            invocation.environment["GROK_HOME"], "/mnt/home/.grok"
        )

    def test_exit_classifiers_separate_model_limits_from_infrastructure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stdout.log").write_text(
                json.dumps(
                    {
                        "type": "result",
                        "result": "API Error: Claude's response exceeded the "
                        "32000 output token maximum.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                classify_claude_exit(root, 1)[0],
                "model_output_exhausted",
            )

            (root / "stdout.log").write_text(
                json.dumps(
                    {
                        "type": "result",
                        "result": "Prompt is too long",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                classify_claude_exit(root, 1)[0],
                "context_window_exhausted",
            )

            (root / "stdout.log").write_text(
                "Incomplete response returned, reason: max_output_tokens\n",
                encoding="utf-8",
            )
            self.assertEqual(
                classify_codex_exit(root, 1)[0],
                "model_output_exhausted",
            )

            (root / "stderr.log").write_text(
                "error_kind: max_tokens_truncation\n",
                encoding="utf-8",
            )
            self.assertEqual(
                classify_grok_exit(root, 1)[0],
                "model_output_exhausted",
            )

    def test_grok_trace_preserves_usage_from_terminal_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = {
                "type": "error",
                "num_turns": 25,
                "usage": {
                    "input_tokens": 80109,
                    "cache_read_input_tokens": 740352,
                    "output_tokens": 18442,
                    "reasoning_tokens": 15029,
                    "total_tokens": 838903,
                },
            }
            (root / "stdout.log").write_text(
                json.dumps(event) + "\n", encoding="utf-8"
            )
            turns, actions, usage = grok_trace(root)
            self.assertEqual(turns, 25)
            self.assertEqual(actions, 0)
            self.assertEqual(usage, event["usage"])

    def test_grok_audit_rejects_tools_manifest_violation(self) -> None:
        run_root, _invocation = self._invocation(
            build_grok_build, "grok-build", "/bin/grok"
        )
        sandbox_events = (
            run_root / "runtime/home/.grok/sandbox-events.jsonl"
        )
        sandbox_events.write_text(
            "\n".join(
                json.dumps(event)
                for event in (
                    {
                        "event_type": "ProfileApplied",
                        "profile": "dataspace-strict",
                        "enforced": True,
                        "read_only_paths": ["/opt/dataspace"],
                    },
                    {
                        "event_type": "FsViolation",
                        "target": "/opt/dataspace/TOOLS.md",
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "tools manifest"):
            validate_grok_backbone(run_root, "xiaomi/mimo-v2.5")

    def test_claude_single_backbone_audit_rejects_subagent_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            events = [
                {
                    "type": "assistant",
                    "message": {"model": "xiaomi/mimo-v2.5"},
                },
                {
                    "type": "assistant",
                    "message": {"model": "claude-opus-4-8"},
                },
            ]
            (run_root / "stdout.log").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "claude-opus-4-8"):
                validate_claude_backbone(run_root, "xiaomi/mimo-v2.5")

    def test_grok_single_backbone_audit_rejects_side_model(self) -> None:
        run_root, _invocation = self._invocation(
            build_grok_build, "grok-build", "/bin/grok"
        )
        runtime_log = run_root / "runtime/home/.grok/logs/unified.jsonl"
        runtime_log.parent.mkdir(parents=True)
        runtime_log.write_text(
            json.dumps(
                {
                    "msg": "backend_search: model switch",
                    "ctx": {"new_model": "xai/grok-4.5"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "xai/grok-4.5"):
            validate_grok_backbone(run_root, "xiaomi/mimo-v2.5")

    def test_claude_and_grok_use_project_local_runtimes(self) -> None:
        cases = (
            (
                "claude-code",
                build_claude_code,
                "/mnt/runtime/claude-code/bin/claude",
                "/mnt/runtime/claude-code",
            ),
            (
                "grok-build",
                build_grok_build,
                "/mnt/runtime/grok-build/bin/grok",
                "/mnt/runtime/grok-build",
            ),
        )
        for key, builder, executable, mount in cases:
            with self.subTest(harness=key):
                temporary = tempfile.TemporaryDirectory()
                self.addCleanup(temporary.cleanup)
                root = Path(temporary.name)
                task = _task(root)
                run_root = root / "run"
                output_root = run_root / "output"
                runtime = root / "runtime"
                runtime.mkdir()
                run_root.mkdir()
                output_root.mkdir()
                config = _config(key, executable, root)
                harness = HarnessConfig(
                    key=key,
                    executable=executable,
                    options={
                        "version": "test-version",
                        "runtime_path": str(runtime),
                        "runtime_mount": mount,
                    },
                )
                with patch.dict(
                    os.environ, {"TEST_GATEWAY_KEY": "test-secret"}
                ):
                    runner = builder(config, harness)
                    invocation = runner.invocation_factory(
                        task, run_root, output_root
                    )

                self.assertIn(str(runtime), invocation.command)
                self.assertIn(mount, invocation.command)
                self.assertIn(executable, invocation.command)
                self.assertEqual(
                    invocation.metadata["runtime_version"], "test-version"
                )

    def test_smolagents_uses_internal_docker_executor(self) -> None:
        run_root, invocation = self._invocation(
            build_smolagents, "smolagents", ""
        )
        resources = json.loads(
            (run_root / "runtime/smolagents_resources.json").read_text(
                encoding="utf-8"
            )
        )
        prompt = (run_root / "runtime/prompt.txt").read_text(encoding="utf-8")
        self.assertIn("/workspace", prompt)
        self.assertIn("/output/prediction.csv", prompt)
        self.assertNotIn("/mnt/workspace", prompt)
        self.assertNotIn("/mnt/output", prompt)
        self.assertTrue(resources["container"].startswith("dataspace-smolagents-"))
        self.assertEqual(resources["container"], resources["network"])
        self.assertEqual(
            invocation.metadata["container_network"], "internal"
        )
        self.assertFalse(invocation.inherit_environment)

    def test_smolagents_container_kwargs_match_docker_sdk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = root / "context"
            output = root / "output"
            context.mkdir()
            output.mkdir()
            options = _container_run_kwargs(
                SimpleNamespace(
                    container_name="dataspace-smolagents-test",
                    network_name="dataspace-smolagents-test",
                    context=context,
                    output=output,
                )
            )

        self.assertNotIn("stop_timeout", options)
        self.assertTrue(options["read_only"])
        self.assertEqual(options["cap_drop"], ["ALL"])
        self.assertEqual(options["network"], "dataspace-smolagents-test")
        self.assertEqual(options["volumes"][str(context)]["mode"], "ro")

    def test_smolagents_trace_counts_executed_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            trace = run_root / "runtime" / "smolagents_trace.json"
            trace.parent.mkdir()
            trace.write_text(
                json.dumps(
                    {
                        "steps": [
                            {},
                            {
                                "step_number": 1,
                                "code_action": "print(1)",
                                "tool_calls": [{"id": "call_1"}],
                            },
                            {
                                "step_number": 2,
                                "code_action": None,
                                "tool_calls": [],
                                "error": {"type": "AgentParsingError"},
                            },
                            {"plan": "inspect, compute, answer"},
                        ],
                        "usage": {"input_tokens": 20, "output_tokens": 5},
                    }
                ),
                encoding="utf-8",
            )
            turns, actions, usage = smolagents_trace(run_root)

        self.assertEqual(turns, 3)
        self.assertEqual(actions, 1)
        self.assertEqual(usage, {"input_tokens": 20, "output_tokens": 5})

    def test_smolagents_interrupted_snapshot_keeps_steps_and_usage(self) -> None:
        class FakeMemory:
            def get_succinct_steps(self):
                return [
                    {"task": "solve"},
                    {
                        "step_number": 1,
                        "code_action": "print(1)",
                        "tool_calls": [{"id": "call_1"}],
                    },
                ]

        class FakeUsage:
            def dict(self):
                return {"input_tokens": 20, "output_tokens": 5}

        class FakeMonitor:
            def get_total_token_counts(self):
                return FakeUsage()

        steps, usage = _agent_snapshot(
            SimpleNamespace(memory=FakeMemory(), monitor=FakeMonitor())
        )

        self.assertEqual(steps[1]["step_number"], 1)
        self.assertEqual(steps[1]["tool_calls"], [{"id": "call_1"}])
        self.assertEqual(
            usage,
            {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
        )

    def test_claude_trace_counts_turns_tools_and_final_usage_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            events = [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "inspect"},
                            {"type": "tool_use", "name": "Bash"},
                        ],
                        "usage": {"input_tokens": 10},
                    },
                },
                {"type": "assistant", "message": {"content": []}},
                {
                    "type": "result",
                    "num_turns": 2,
                    "usage": {"input_tokens": 20, "output_tokens": 5},
                },
            ]
            (run_root / "stdout.log").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            turns, actions, usage = claude_trace(run_root)

        self.assertEqual(turns, 2)
        self.assertEqual(actions, 1)
        self.assertEqual(usage, {"input_tokens": 20, "output_tokens": 5})

    def test_codex_trace_counts_each_tool_item_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            events = [
                {
                    "type": "item.started",
                    "item": {
                        "id": "item_1",
                        "type": "command_execution",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_1",
                        "type": "command_execution",
                    },
                },
                {
                    "type": "item.started",
                    "item": {"id": "item_2", "type": "file_change"},
                },
                {
                    "type": "item.completed",
                    "item": {"id": "item_2", "type": "file_change"},
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 20, "output_tokens": 5},
                },
            ]
            (run_root / "stdout.log").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            turns, actions, usage = codex_trace(run_root)

        self.assertEqual(turns, 1)
        self.assertEqual(actions, 2)
        self.assertEqual(usage, {"input_tokens": 20, "output_tokens": 5})

    def test_grok_trace_uses_end_event_and_runtime_tool_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            events = [
                {"type": "text", "data": "Working"},
                {"type": "thought", "data": "Inspect the workspace"},
                {
                    "type": "end",
                    "num_turns": 3,
                    "usage": {
                        "input_tokens": 20,
                        "cache_read_input_tokens": 30,
                        "output_tokens": 5,
                        "total_tokens": 55,
                    },
                },
            ]
            (run_root / "stdout.log").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            runtime_log = (
                run_root / "runtime/home/.grok/logs/unified.jsonl"
            )
            runtime_log.parent.mkdir(parents=True)
            runtime_events = [
                {"msg": "shell.turn.inference_done"},
                {"msg": "shell.tool.exec_done", "ctx": {"success": False}},
                {"msg": "shell.turn.inference_done"},
                {"msg": "shell.tool.exec_done", "ctx": {"success": True}},
                {"msg": "shell.turn.inference_done"},
            ]
            runtime_log.write_text(
                "\n".join(json.dumps(event) for event in runtime_events) + "\n",
                encoding="utf-8",
            )
            turns, actions, usage = grok_trace(run_root)

        self.assertEqual(turns, 3)
        self.assertEqual(actions, 2)
        self.assertEqual(
            usage,
            {
                "input_tokens": 20,
                "cache_read_input_tokens": 30,
                "output_tokens": 5,
                "total_tokens": 55,
            },
        )


if __name__ == "__main__":
    unittest.main()
