import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from minisweagent import Environment
from minisweagent.utils.log import logger as ms_logger

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    MCPConfig,
    OpenHandsConfig,
    SandboxConfig,
)
from openhands.core.config.llm_config import LLMConfig
from openhands.core.schema import AgentState
from openhands.events import EventSource, EventStream, EventStreamSubscriber
from openhands.events.action import MessageAction
from openhands.events.action.commands import CmdRunAction
from openhands.events.action.agent import AgentFinishAction
from openhands.events.observation import (
    AgentThinkObservation,
    CmdOutputObservation,
    ErrorObservation,
    Observation,
)
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.base import Runtime
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.local import LocalFileStore

from minisweagent.utils.paths import get_repo_tmp


class _MiniRuntime(Runtime):
    """Minimal runtime that forwards CmdRunAction to mini-swe-agent Environment."""

    def __init__(self, env: Environment, *args, **kwargs):
        self.env = env
        super().__init__(*args, **kwargs)

    async def connect(self) -> None:  # pragma: no cover - no-op for this adapter
        self._runtime_initialized = True

    def close(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.event_stream.unsubscribe(EventStreamSubscriber.RUNTIME, self.sid)
        except Exception:
            pass
        try:
            cleanup = getattr(self.env, "cleanup", None)
            if callable(cleanup):
                cleanup()
        except Exception:
            pass

    def get_mcp_config(self, extra_stdio_servers: list | None = None) -> MCPConfig:
        return MCPConfig()

    def run(self, action: CmdRunAction) -> Observation:
        cwd = action.cwd or ""
        try:
            result = self.env.execute(action.command, cwd=cwd)
            output = result.get("output", "")
            exit_code = result.get("returncode", result.get("exit_code", 0))
            metadata = {"exit_code": exit_code, "working_dir": cwd}
            return CmdOutputObservation(
                content=output,
                command=action.command,
                metadata=metadata,
            )
        except Exception as e:  # pragma: no cover
            return ErrorObservation(f"Runtime execution failed: {e}")

    # Unsupported operations fall back to ErrorObservation to keep agent moving
    def run_ipython(self, action) -> Observation:
        return ErrorObservation("IPython not supported in mini runtime.")

    def read(self, action) -> Observation:
        return ErrorObservation("File read not supported in mini runtime.")

    def write(self, action) -> Observation:
        return ErrorObservation("File write not supported in mini runtime.")

    def edit(self, action) -> Observation:
        return ErrorObservation("File edit not supported in mini runtime.")

    def browse(self, action) -> Observation:
        return ErrorObservation("Browse not supported in mini runtime.")

    def browse_interactive(self, action) -> Observation:
        return ErrorObservation("Browse interactive not supported in mini runtime.")

    async def call_tool_mcp(self, action) -> Observation:
        return ErrorObservation("MCP not supported in mini runtime.")

    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        raise NotImplementedError("copy_to not supported in mini runtime.")

    def list_files(self, path: str | None = None) -> list[str]:
        return []

    def copy_from(self, path: str):
        raise NotImplementedError("copy_from not supported in mini runtime.")


@dataclass
class CodeActResult:
    exit_status: str
    result: str
    steps_path: str | None = None
    history_path: str | None = None


class CodeActRunner:
    """Runs CodeActAgent decisions while executing commands via mini-swe-agent Environment."""

    def __init__(
        self,
        env: Environment,
        llm_config: dict[str, Any],
        *,
        max_steps: int = 100,
        file_store_root: str | None = None,
        run_id: str | None = None,
    ):
        self.env = env
        self.llm_config = llm_config or {}
        self.max_steps = max_steps
        self.file_store_root = file_store_root or str(get_repo_tmp() / "codeact_mswea_store")
        self.run_id = run_id

    def _collect_patch(self) -> str:
        """Collect working-tree diff from the task repo inside the container."""
        try:
            resp = self.env.execute("git -C /testbed diff")
        except Exception as e:  # pragma: no cover
            ms_logger.error(f"[CodeActRunner] failed to collect patch: {e}")
            return ""
        rc = resp.get("returncode", resp.get("exit_code", 0))
        output = resp.get("output", "")
        if rc != 0:
            ms_logger.error(f"[CodeActRunner] git diff failed rc={rc}")
            return ""
        if output and output.strip():
            ms_logger.info(f"[CodeActRunner] collected git diff patch ({len(output)} chars)")
            return output
        return ""

    def _make_config(self) -> OpenHandsConfig:
        llm_data = dict(self.llm_config)
        llm_data.setdefault("custom_llm_provider", "openai")
        llm_data.setdefault("timeout", 120)
        agent_cfg = AgentConfig(
            enable_browsing=False,
            enable_jupyter=False,
            enable_editor=False,
            enable_llm_editor=False,
            enable_plan_mode=False,
            enable_condensation_request=False,
            enable_mcp=False,
            runtime="cli",
        )
        agent_cfg.model_post_init(None)
        cfg = OpenHandsConfig(
            llms={"llm": LLMConfig(**llm_data)},
            agents={"agent": agent_cfg},
            default_agent="agent",
            sandbox=SandboxConfig(),
        )
        return cfg

    def _build_runtime(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        llm_registry: LLMRegistry,
        sid: str,
    ) -> _MiniRuntime:
        runtime = _MiniRuntime(
            env=self.env,
            config=config,
            event_stream=event_stream,
            llm_registry=llm_registry,
            sid=sid,
            headless_mode=True,
        )
        return runtime

    def run_instance(self, task: str) -> CodeActResult:
        sid = self.run_id or f"codeact-{uuid.uuid4().hex[:8]}"
        file_store_path = os.path.join(self.file_store_root, sid)
        os.makedirs(file_store_path, exist_ok=True)
        steps_path = os.path.join(file_store_path, "steps.jsonl")
        history_path = os.path.join(file_store_path, "history.jsonl")
        file_store = LocalFileStore(file_store_path)

        config = self._make_config()
        llm_registry = LLMRegistry(config=config, agent_cls="agent")
        event_stream = EventStream(sid=sid, file_store=file_store, user_id=None)
        conversation_stats = ConversationStats(file_store, sid, None)
        llm_registry.subscribe(conversation_stats.register_llm)
        runtime = self._build_runtime(
            config=config,
            event_stream=event_stream,
            llm_registry=llm_registry,
            sid=sid,
        )

        agent_config = config.get_agent_config()
        agent = CodeActAgent(config=agent_config, llm_registry=llm_registry)

        # Build initial state/history (synchronous loop; no asyncio)
        state = State(session_id=sid, conversation_stats=conversation_stats)
        state.agent_state = AgentState.RUNNING
        system_msg = agent.get_system_message()
        if system_msg:
            system_msg._source = EventSource.AGENT  # type: ignore[attr-defined]
            state.history.append(system_msg)
        user_msg = MessageAction(content=task, wait_for_response=False)
        user_msg._source = EventSource.USER  # type: ignore[attr-defined]
        state.history.append(user_msg)

        ms_logger.info(
            f"[CodeActRunner] start run_instance sid={sid} model={agent.llm.config.model} max_steps={self.max_steps}"
        )

        exit_status = "timeout"
        result = ""
        for step_idx in range(self.max_steps):
            ms_logger.info(f"[CodeActRunner] iter={step_idx+1} sid={sid} calling agent.step")
            try:
                action = agent.step(state)
            except Exception as e:  # LLM or parsing failure
                exit_status = "error"
                result = f"Agent step failed: {e}"
                break

            ms_logger.info(f"[CodeActRunner] iter={step_idx+1} sid={sid} got action {type(action).__name__}")

            if isinstance(action, AgentFinishAction):
                exit_status = "finished"
                result = ""
                if isinstance(action.outputs, dict) and action.outputs.get("content"):
                    result = action.outputs.get("content", "") or ""
                if not result:
                    result = getattr(action, "final_thought", "") or ""
                if not result:
                    result = getattr(action, "thought", "") or ""
                state.history.append(action)
                break

            state.history.append(action)

            if isinstance(action, CmdRunAction):
                obs = runtime.run(action)
            elif hasattr(action, "action") and getattr(action, "action", "") == "think":
                obs = AgentThinkObservation(action.thought)
            else:
                obs = ErrorObservation(f"Action type {type(action).__name__} not supported in CodeActRunner")

            obs._source = EventSource.ENVIRONMENT  # type: ignore[attr-defined]
            state.history.append(obs)
            if isinstance(obs, CmdOutputObservation):
                ms_logger.info(
                    f"[CodeActRunner] iter={step_idx+1} sid={sid} obs CmdOutput exit={obs.exit_code} len={len(obs.content)}"
                )
            elif isinstance(obs, ErrorObservation):
                ms_logger.error(f"[CodeActRunner] iter={step_idx+1} sid={sid} obs Error: {obs.content}")
            else:
                ms_logger.info(f"[CodeActRunner] iter={step_idx+1} sid={sid} obs {type(obs).__name__}")

            # persist step
            step_rec = {
                "iter": step_idx + 1,
                "action_type": type(action).__name__,
                "action": getattr(action, "command", None) or getattr(action, "thought", None) or getattr(action, "final_thought", None),
                "observation_type": type(obs).__name__,
                "exit_code": getattr(obs, "exit_code", None),
                "obs_len": len(getattr(obs, "content", "") or ""),
            }
            try:
                with open(steps_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(step_rec, ensure_ascii=False) + "\n")
            except Exception as e:
                ms_logger.error(f"[CodeActRunner] failed to write steps log: {e}")

            # Track iterations
            state.iteration_flag.current_value += 1

            if isinstance(obs, ErrorObservation):
                exit_status = "error"
                result = obs.content
                break
        else:
            exit_status = "timeout"
            result = "Agent reached max steps without finishing"

        ms_logger.info(f"[CodeActRunner] end run_instance sid={sid} status={exit_status}")

        # Prefer returning the actual code diff when available.
        if exit_status == "finished":
            patch = self._collect_patch()
            if patch:
                result = patch

        # persist full history (system + user + actions + observations)
        try:
            with open(history_path, "w", encoding="utf-8") as f:
                for ev in state.history:
                    rec = {
                        "type": type(ev).__name__,
                        "source": getattr(ev, "source", None),
                        "content": getattr(ev, "content", None),
                        "command": getattr(ev, "command", None),
                        "thought": getattr(ev, "thought", None),
                        "outputs": getattr(ev, "outputs", None),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            ms_logger.error(f"[CodeActRunner] failed to write history log: {e}")

        try:
            runtime.close()
            event_stream.close()
        except Exception:
            pass

        return CodeActResult(
            exit_status=exit_status,
            result=result,
            steps_path=steps_path,
            history_path=history_path,
        )
