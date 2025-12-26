import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from minisweagent import Environment
from minisweagent.integrations.openhands_runtime import OpenHandsCompatRuntime
from minisweagent.utils.log import logger as ms_logger

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, OpenHandsConfig, SandboxConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
    LLMMalformedActionError,
    LLMNoActionError,
    LLMResponseError,
)
from openhands.core.schema import AgentState
from openhands.events import EventSource, EventStream, EventStreamSubscriber
from openhands.events.action import (
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    MessageAction,
    TaskTrackingAction,
)
from openhands.events.action.browse import BrowseInteractiveAction, BrowseURLAction
from openhands.events.action.commands import CmdRunAction, IPythonRunCellAction
from openhands.events.action.mcp import MCPAction
from openhands.events.action.agent import AgentFinishAction, AgentThinkAction
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.observation import (
    AgentThinkObservation,
    CmdOutputObservation,
    ErrorObservation,
    FileEditObservation,
    FileReadObservation,
    FileWriteObservation,
    Observation,
)
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.base import Runtime
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.local import LocalFileStore

from minisweagent.utils.paths import get_repo_tmp


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
        """Collect working-tree diff from the task repo inside the container.

        We stage all changes (including untracked files) to mirror the manual
        `git add -A && git diff --cached` flow used by the CLI instructions and
        then reset the index so we don't alter the working tree state.
        """
        repo_path = "/testbed"
        try:
            status = self.env.execute(f"git -C {repo_path} status --porcelain")
        except Exception as e:  # pragma: no cover
            ms_logger.error(f"[CodeActRunner] failed to collect patch: {e}")
            return ""
        status_rc = status.get("returncode", status.get("exit_code", 0))
        changes = status.get("output", "")
        if status_rc != 0:
            ms_logger.error(f"[CodeActRunner] git status failed rc={status_rc}")
            return ""
        if not changes.strip():
            ms_logger.info("[CodeActRunner] no git changes to collect")
            return ""

        try:
            add_resp = self.env.execute(f"git -C {repo_path} add -A")
            add_rc = add_resp.get("returncode", add_resp.get("exit_code", 0))
            if add_rc != 0:
                ms_logger.error(f"[CodeActRunner] git add failed rc={add_rc}")
                return ""

            diff_resp = self.env.execute(f"git -C {repo_path} diff --cached")
            diff_rc = diff_resp.get("returncode", diff_resp.get("exit_code", 0))
            patch = diff_resp.get("output", "")
            if diff_rc != 0:
                ms_logger.error(f"[CodeActRunner] git diff --cached failed rc={diff_rc}")
                return ""
            if patch and patch.strip():
                ms_logger.info(f"[CodeActRunner] collected git diff patch ({len(patch)} chars)")
                return patch
            ms_logger.error("[CodeActRunner] staged changes detected but diff was empty")
            return ""
        finally:
            try:
                self.env.execute(f"git -C {repo_path} reset")
            except Exception:
                pass

    def _make_config(self, llm_data: dict[str, Any] | None = None) -> OpenHandsConfig:
        llm_data = dict(llm_data or self.llm_config)
        llm_data.setdefault("custom_llm_provider", "openai")
        llm_data.setdefault("timeout", 120)
        # Force native tool calling when supported to preserve assistant/tool history
        llm_data.setdefault("native_tool_calling", True)
        agent_cfg = AgentConfig(
            enable_jupyter=False,
            enable_browsing=os.environ.get("RUN_WITH_BROWSING", "false").lower()
            == "true",
            enable_llm_editor=False,
            enable_mcp=False,
            enable_prompt_extensions=False,
            enable_plan_mode=False,
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
    ) -> OpenHandsCompatRuntime:
        runtime = OpenHandsCompatRuntime(
            env=self.env,
            config=config,
            event_stream=event_stream,
            llm_registry=llm_registry,
            sid=sid,
            headless_mode=True,
        )
        return runtime

    def run_instance(self, task: str, progress_callback: Callable[[str], None] | None = None) -> CodeActResult:
        sid = self.run_id or f"codeact-{uuid.uuid4().hex[:8]}"
        file_store_path = os.path.join(self.file_store_root, sid)
        os.makedirs(file_store_path, exist_ok=True)
        steps_path = os.path.join(file_store_path, "steps.jsonl")
        history_path = os.path.join(file_store_path, "history.jsonl")
        iter_log_path = os.path.join(file_store_path, "runner.log")
        file_store = LocalFileStore(file_store_path)

        llm_data = dict(self.llm_config)
        if llm_data.get("log_completions"):
            llm_data["log_completions_folder"] = os.path.join(file_store_path, "llm_completions")
            try:
                os.makedirs(llm_data["log_completions_folder"], exist_ok=True)
            except Exception:
                pass

        config = self._make_config(llm_data)
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

        status_prefix = f"CodeAct sid={sid}"
        ms_logger.info(
            f"[CodeActRunner] start run_instance sid={sid} model={agent.llm.config.model} max_steps={self.max_steps}"
        )
        if progress_callback:
            try:
                progress_callback("CodeAct: starting")
            except Exception:
                pass
        try:
            with open(iter_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"start sid={sid} model={agent.llm.config.model} max_steps={self.max_steps}\n"
                )
        except Exception:
            pass

        exit_status = "timeout"
        result = ""
        for step_idx in range(self.max_steps):
            ms_logger.info(f"[CodeActRunner] iter={step_idx+1} sid={sid} calling agent.step")
            if progress_callback:
                try:
                    progress_callback(f"CodeAct iter {step_idx+1}")
                except Exception:
                    pass
            try:
                with open(iter_log_path, "a", encoding="utf-8") as f:
                    f.write(f"iter={step_idx+1} calling agent.step\n")
            except Exception:
                pass
            try:
                action = agent.step(state)
            except (
                LLMMalformedActionError,
                LLMNoActionError,
                LLMResponseError,
                FunctionCallValidationError,
                FunctionCallNotExistsError,
            ) as e:
                obs = ErrorObservation(content=str(e))
                obs._source = EventSource.AGENT  # type: ignore[attr-defined]
                state.history.append(obs)
                ms_logger.error(
                    f"[CodeActRunner] iter={step_idx+1} sid={sid} agent.step tool error: {e}"
                )
                try:
                    with open(iter_log_path, "a", encoding="utf-8") as f:
                        f.write(f"iter={step_idx+1} agent_step_tool_error={e!r}\n")
                except Exception:
                    pass
                step_rec = {
                    "iter": step_idx + 1,
                    "action_type": "AgentStepError",
                    "action": str(e),
                    "observation_type": type(obs).__name__,
                    "exit_code": None,
                    "obs_len": len(obs.content or ""),
                }
                try:
                    with open(steps_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(step_rec, ensure_ascii=False) + "\n")
                except Exception as e:
                    ms_logger.error(f"[CodeActRunner] failed to write steps log: {e}")
                state.iteration_flag.current_value += 1
                continue
            except Exception as e:  # LLM or parsing failure
                exit_status = "error"
                result = f"Agent step failed: {e}"
                break

            ms_logger.info(f"[CodeActRunner] iter={step_idx+1} sid={sid} got action {type(action).__name__}")
            try:
                with open(iter_log_path, "a", encoding="utf-8") as f:
                    f.write(f"iter={step_idx+1} action={type(action).__name__}\n")
            except Exception:
                pass

            # Ensure agent-produced actions are marked with source=agent so ConversationMemory keeps them
            if getattr(action, "source", None) is None:
                try:
                    # Some actions expose _source instead of a writable property
                    if hasattr(action, "_source"):
                        setattr(action, "_source", EventSource.AGENT)
                    else:
                        setattr(action, "source", EventSource.AGENT)
                except Exception:
                    try:
                        setattr(action, "_source", "agent")
                    except Exception:
                        pass

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
                try:
                    obs.tool_call_metadata = getattr(action, "tool_call_metadata", None)
                except Exception:
                    pass
            elif isinstance(action, FileReadAction):
                obs = runtime.read(action)
                try:
                    obs.tool_call_metadata = getattr(action, "tool_call_metadata", None)
                except Exception:
                    pass
            elif isinstance(action, FileWriteAction):
                obs = runtime.write(action)
                try:
                    obs.tool_call_metadata = getattr(action, "tool_call_metadata", None)
                except Exception:
                    pass
            elif isinstance(action, FileEditAction):
                obs = runtime.edit(action)
                try:
                    obs.tool_call_metadata = getattr(action, "tool_call_metadata", None)
                except Exception:
                    pass
            elif isinstance(action, IPythonRunCellAction):
                obs = runtime.run_ipython(action)
                try:
                    obs.tool_call_metadata = getattr(action, "tool_call_metadata", None)
                except Exception:
                    pass
            elif isinstance(action, BrowseURLAction):
                obs = runtime.browse(action)
                try:
                    obs.tool_call_metadata = getattr(action, "tool_call_metadata", None)
                except Exception:
                    pass
            elif isinstance(action, BrowseInteractiveAction):
                obs = runtime.browse_interactive(action)
                try:
                    obs.tool_call_metadata = getattr(action, "tool_call_metadata", None)
                except Exception:
                    pass
            elif isinstance(action, MCPAction):
                try:
                    obs = asyncio.run(runtime.call_tool_mcp(action))
                except RuntimeError as e:
                    obs = ErrorObservation(f"MCP call failed: {e}")
                try:
                    obs.tool_call_metadata = getattr(action, "tool_call_metadata", None)
                except Exception:
                    pass
            elif isinstance(action, AgentThinkAction):
                obs = runtime.run_action(action)
                try:
                    obs.tool_call_metadata = getattr(action, "tool_call_metadata", None)
                except Exception:
                    pass
            elif isinstance(action, TaskTrackingAction):
                obs = runtime.run_action(action)
                try:
                    obs.tool_call_metadata = getattr(action, "tool_call_metadata", None)
                except Exception:
                    pass
            elif isinstance(action, MessageAction):
                # Non-tool assistant messages can happen if the model skips tool calls.
                # Treat as a thought so the loop can continue.
                msg = action.content or ""
                obs = AgentThinkObservation(msg)
                ms_logger.info(
                    f"[CodeActRunner] iter={step_idx+1} sid={sid} MessageAction -> AgentThinkObservation"
                )
                try:
                    with open(iter_log_path, "a", encoding="utf-8") as f:
                        f.write(f"iter={step_idx+1} message_action_content={msg[:200]!r}\n")
                except Exception:
                    pass
            elif hasattr(action, "action") and getattr(action, "action", "") == "think":
                obs = AgentThinkObservation("Your thought has been logged.")
                try:
                    obs.tool_call_metadata = getattr(action, "tool_call_metadata", None)
                except Exception:
                    pass
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
            try:
                with open(iter_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"iter={step_idx+1} obs={type(obs).__name__} exit={getattr(obs, 'exit_code', None)} len={len(getattr(obs, 'content', '') or '')}\n"
                    )
            except Exception:
                pass

            # persist step
            step_rec = {
                "iter": step_idx + 1,
                "action_type": type(action).__name__,
                "action": (
                    getattr(action, "command", None)
                    or getattr(action, "thought", None)
                    or getattr(action, "final_thought", None)
                    or getattr(action, "content", None)
                ),
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
