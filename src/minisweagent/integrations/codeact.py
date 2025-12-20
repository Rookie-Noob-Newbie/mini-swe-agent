import asyncio
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from minisweagent import Environment

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    MCPConfig,
    OpenHandsConfig,
    SandboxConfig,
)
from openhands.core.config.llm_config import LLMConfig
from openhands.core.logger import openhands_logger as oh_logger
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

    def _make_config(self) -> OpenHandsConfig:
        llm_data = dict(self.llm_config)
        llm_data.setdefault("custom_llm_provider", "openai")
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

        oh_logger.info(
            f"[CodeActRunner] start run_instance sid={sid} model={agent.llm.config.model} max_steps={self.max_steps}"
        )

        exit_status = "timeout"
        result = ""
        for _ in range(self.max_steps):
            try:
                action = agent.step(state)
            except Exception as e:  # LLM or parsing failure
                exit_status = "error"
                result = f"Agent step failed: {e}"
                break

            if isinstance(action, AgentFinishAction):
                exit_status = "finished"
                result = (
                    action.outputs.get("content")
                    if isinstance(action.outputs, dict)
                    else action.final_thought or ""
                )
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

            # Track iterations
            state.iteration_flag.current_value += 1

            if isinstance(obs, ErrorObservation):
                exit_status = "error"
                result = obs.content
                break
        else:
            exit_status = "timeout"
            result = "Agent reached max steps without finishing"

        oh_logger.info(f"[CodeActRunner] end run_instance sid={sid} status={exit_status}")

        try:
            runtime.close()
            event_stream.close()
        except Exception:
            pass

        return CodeActResult(exit_status=exit_status, result=result)
