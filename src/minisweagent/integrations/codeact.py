import asyncio
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from minisweagent import Environment

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.agent_controller import AgentController
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    LLMConfig,
    MCPConfig,
    OpenHandsConfig,
    SandboxConfig,
)
from openhands.core.logger import openhands_logger as oh_logger
from openhands.core.schema import AgentState
from openhands.events import EventSource, EventStream, EventStreamSubscriber
from openhands.events.action import MessageAction
from openhands.events.action.commands import CmdRunAction
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    Observation,
)
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.base import Runtime
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.local import LocalFileStore


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
        self.llm_config = llm_config
        self.max_steps = max_steps
        self.file_store_root = file_store_root or os.path.join(
            tempfile.gettempdir(), "codeact_mswea_store"
        )
        self.run_id = run_id

    def _make_config(self) -> OpenHandsConfig:
        agent_cfg = AgentConfig(
            enable_browsing=False,
            enable_jupyter=False,
            enable_editor=False,
            enable_llm_editor=False,
            enable_plan_mode=False,
            enable_condensation_request=False,
            enable_mcp=False,
            runtime="cli",
            max_iterations=self.max_steps,
        )
        cfg = OpenHandsConfig(
            llms={"llm": LLMConfig(**self.llm_config)},
            agents={"agent": agent_cfg},
            default_agent="agent",
            sandbox=SandboxConfig(),
            max_iterations=self.max_steps,
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

    def _wait_until_done(self, controller: AgentController, deadline: float) -> None:
        while time.time() < deadline:
            state = controller.get_agent_state()
            if state in (
                AgentState.FINISHED,
                AgentState.ERROR,
                AgentState.REJECTED,
            ):
                return
            time.sleep(0.1)
        oh_logger.warning("CodeActRunner hit wall-clock timeout waiting for completion.")

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

        agent = CodeActAgent(config=config.get_agent_config(), llm_registry=llm_registry)

        controller = AgentController(
            agent=agent,
            event_stream=event_stream,
            conversation_stats=conversation_stats,
            iteration_delta=self.max_steps,
            budget_per_task_delta=None,
            agent_to_llm_config=config.get_agent_to_llm_config_map(),
            agent_configs=config.get_agent_configs(),
            sid=sid,
            headless_mode=True,
            status_callback=None,
        )

        finished = {"status": "", "result": ""}
        finished_lock = threading.Lock()

        def _observer(event):
            from openhands.events.action import AgentFinishAction
            from openhands.events.observation import AgentStateChangedObservation

            if isinstance(event, AgentFinishAction):
                with finished_lock:
                    finished["status"] = "finished"
                    finished["result"] = (
                        event.outputs.get("content")
                        if isinstance(event.outputs, dict)
                        else event.final_thought or ""
                    )
            elif isinstance(event, AgentStateChangedObservation):
                if event.agent_state == AgentState.ERROR:
                    with finished_lock:
                        finished["status"] = "error"
                        finished["result"] = (
                            controller.state.last_error or "Agent error"
                        )

        event_stream.subscribe(EventStreamSubscriber.MAIN, _observer, f"observer-{sid}")

        # Kick off with initial user message
        msg = MessageAction(content=task, wait_for_response=False)
        event_stream.add_event(msg, EventSource.USER)

        # Wait for completion or timeout
        self._wait_until_done(controller, time.time() + 60 * 5)

        with finished_lock:
            status = finished["status"] or controller.get_agent_state().name.lower()
            result = finished["result"] or controller.state.last_error or ""

        try:
            runtime.close()
            event_stream.close()
        except Exception:
            pass

        return CodeActResult(exit_status=status, result=result)
