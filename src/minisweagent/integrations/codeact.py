import asyncio
import json
import os
import shlex
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
from openhands.events.serialization.event import event_to_dict
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.base import Runtime
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.local import LocalFileStore

from minisweagent.utils.paths import get_repo_tmp


@dataclass
class CodeActResult:
    exit_status: str
    result: str
    steps_path: str
    history_path: str


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
        repo_path: str | None = None,
        base_commit: str | None = None,
    ):
        self.env = env
        self.llm_config = llm_config or {}
        self.max_steps = max_steps
        self.file_store_root = file_store_root or str(get_repo_tmp() / "codeact_mswea_store")
        self.run_id = run_id
        self.repo_path = repo_path or "/testbed"
        self.base_commit = base_commit

    @staticmethod
    def _remove_binary_diffs(patch_text: str) -> str:
        lines = patch_text.splitlines()
        cleaned_lines: list[str] = []
        block: list[str] = []
        is_binary_block = False

        for line in lines:
            if line.startswith("diff --git "):
                if block and not is_binary_block:
                    cleaned_lines.extend(block)
                block = [line]
                is_binary_block = False
            elif "Binary files" in line:
                is_binary_block = True
                block.append(line)
            else:
                block.append(line)

        if block and not is_binary_block:
            cleaned_lines.extend(block)
        return "\n".join(cleaned_lines)

    @staticmethod
    def _remove_binary_files_command() -> str:
        return """
        for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
            if [ -f "$file" ] && (file "$file" | grep -q "executable" || git check-attr binary "$file" | grep -q "binary: set"); then
                git rm -f "$file" 2>/dev/null || rm -f "$file"
                echo "Removed: $file"
            fi
        done
        """.strip()

    def _collect_patch(self) -> str:
        """Collect working-tree diff from the task repo inside the container.

        We stage all changes (including untracked files), remove binary files from
        staging, generate a patch against the base commit, and then reset the index
        so we don't alter the working tree state.
        """
        repo_path = self.repo_path
        repo_path_q = shlex.quote(repo_path)
        base_commit = self.base_commit
        if not base_commit:
            raise RuntimeError("base_commit is required to collect patch")
        try:
            try:
                self.env.execute(f"git -C {repo_path_q} config --global core.pager \"\"")
                self.env.execute(f"git -C {repo_path_q} config --global diff.binary false")
            except Exception:
                pass

            find_cmd = (
                f"find {repo_path_q} -type d -name .git -not -path "
                f"{shlex.quote(repo_path + '/.git')}"
            )
            find_resp = self.env.execute(find_cmd)
            find_rc = find_resp.get("returncode", find_resp.get("exit_code", 0))
            if find_rc == 0:
                git_dirs = [
                    p for p in (find_resp.get("output", "") or "").splitlines() if p.strip()
                ]
                for git_dir in git_dirs:
                    rm_resp = self.env.execute(f"rm -rf {shlex.quote(git_dir)}")
                    rm_rc = rm_resp.get("returncode", rm_resp.get("exit_code", 0))
                    if rm_rc != 0:
                        ms_logger.error(f"[CodeActRunner] failed to remove git dir {git_dir} rc={rm_rc}")
            else:
                ms_logger.error(f"[CodeActRunner] find .git dirs failed rc={find_rc}")

            add_resp = self.env.execute(f"git -C {repo_path_q} add -A")
            add_rc = add_resp.get("returncode", add_resp.get("exit_code", 0))
            if add_rc != 0:
                ms_logger.error(f"[CodeActRunner] git add failed rc={add_rc}")
                return ""

            remove_binary_cmd = self._remove_binary_files_command()
            bin_resp = self.env.execute(f"cd {repo_path_q} && {remove_binary_cmd}")
            bin_rc = bin_resp.get("returncode", bin_resp.get("exit_code", 0))
            if bin_rc != 0:
                ms_logger.error(f"[CodeActRunner] remove binary files failed rc={bin_rc}")

            n_retries = 0
            git_patch = None
            while n_retries < 5:
                timeout = max(300 + 100 * n_retries, 600)
                n_retries += 1
                diff_cmd = f"git diff --no-color --cached {shlex.quote(base_commit)} > patch.diff"
                diff_resp = self.env.execute(f"cd {repo_path_q} && {diff_cmd}", timeout=timeout)
                diff_rc = diff_resp.get("returncode", diff_resp.get("exit_code", 0))
                if diff_rc != 0:
                    ms_logger.info("[CodeActRunner] Failed to get git diff, retrying...")
                    time.sleep(10)
                    continue

                read_cmd = (
                    f"cd {repo_path_q} && python - <<'PY'\n"
                    "import sys\n"
                    "try:\n"
                    "    with open('patch.diff', 'r', encoding='utf-8') as f:\n"
                    "        sys.stdout.write(f.read())\n"
                    "except UnicodeDecodeError:\n"
                    "    sys.stdout.write('File could not be decoded as utf-8')\n"
                    "    sys.exit(1)\n"
                    "except Exception as e:\n"
                    "    sys.stdout.write(str(e))\n"
                    "    sys.exit(1)\n"
                    "PY"
                )
                read_resp = self.env.execute(read_cmd, timeout=timeout)
                read_rc = read_resp.get("returncode", read_resp.get("exit_code", 0))
                if read_rc == 0:
                    git_patch = read_resp.get("output", "")
                    break

                read_err = read_resp.get("output", "")
                if "File could not be decoded as utf-8" in read_err:
                    patch_resp = self.env.execute(
                        f"cd {repo_path_q} && cat patch.diff",
                        timeout=timeout,
                    )
                    patch_rc = patch_resp.get("returncode", patch_resp.get("exit_code", 0))
                    if patch_rc == 0:
                        git_patch = patch_resp.get("output", "")
                        break
                    ms_logger.error(f"[CodeActRunner] cat patch.diff failed rc={patch_rc}")
                else:
                    ms_logger.error(f"[CodeActRunner] Failed to read patch.diff: {read_err}")

                time.sleep(10)

            if git_patch is None:
                raise RuntimeError("Failed to get git diff (None)")

            git_patch = self._remove_binary_diffs(git_patch)
            ms_logger.info(f"[CodeActRunner] collected git diff patch ({len(git_patch)} chars)")
            return git_patch
        finally:
            try:
                self.env.execute(f"git -C {repo_path_q} reset")
            except Exception:
                pass

    def _fake_user_response(self, state: State) -> str:
        msg = (
            "Please continue working on the task on whatever approach you think is suitable.\n"
            "When you think you have solved the question, please use the finish tool and include your final answer in the message parameter of the finish tool.\n"
            "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n"
        )
        user_msgs = [
            event
            for event in state.history
            if isinstance(event, MessageAction) and event.source == EventSource.USER
        ]
        if len(user_msgs) >= 2:
            return msg + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
        return msg

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
                msg = action.content or ""
                fake_msg = self._fake_user_response(state)
                user_msg = MessageAction(content=fake_msg, wait_for_response=False)
                user_msg._source = EventSource.USER  # type: ignore[attr-defined]
                state.history.append(user_msg)
                ms_logger.info(
                    f"[CodeActRunner] iter={step_idx+1} sid={sid} MessageAction -> fake user response"
                )
                try:
                    with open(iter_log_path, "a", encoding="utf-8") as f:
                        f.write(f"iter={step_idx+1} message_action_content={msg[:200]!r}\n")
                        f.write(f"iter={step_idx+1} fake_user_response_len={len(fake_msg)}\n")
                except Exception:
                    pass

                action_summary = msg
                try:
                    action_payload = event_to_dict(action)
                    action_args = action_payload.get("args")
                except Exception:
                    action_args = None
                step_rec = {
                    "iter": step_idx + 1,
                    "action_type": type(action).__name__,
                    "action": action_summary,
                    "action_message": getattr(action, "message", None),
                    "action_thought": getattr(action, "thought", None),
                    "action_args": action_args,
                    "observation_type": "UserMessageAction",
                    "exit_code": None,
                    "obs_len": len(fake_msg),
                }
                try:
                    with open(steps_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(step_rec, ensure_ascii=False) + "\n")
                except Exception as e:
                    ms_logger.error(f"[CodeActRunner] failed to write steps log: {e}")
                state.iteration_flag.current_value += 1
                continue
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
            action_summary = None
            if isinstance(action, CmdRunAction):
                action_summary = action.command
            elif isinstance(action, FileReadAction):
                action_summary = action.path
            elif isinstance(action, FileWriteAction):
                action_summary = action.path
            elif isinstance(action, FileEditAction):
                action_summary = f"{action.command} {action.path}".strip() or action.path
            elif isinstance(action, IPythonRunCellAction):
                action_summary = action.code
            elif isinstance(action, BrowseURLAction):
                action_summary = action.url
            elif isinstance(action, BrowseInteractiveAction):
                action_summary = action.browser_actions
            elif isinstance(action, MCPAction):
                action_summary = f"{action.name} {action.arguments}"
            elif isinstance(action, AgentFinishAction):
                action_summary = (
                    getattr(action, "final_thought", None)
                    or getattr(action, "thought", None)
                    or (action.outputs or {}).get("content")
                )
            elif isinstance(action, AgentThinkAction):
                action_summary = action.thought
            elif isinstance(action, TaskTrackingAction):
                action_summary = action.command
            elif isinstance(action, MessageAction):
                action_summary = action.content
            else:
                action_summary = (
                    getattr(action, "command", None)
                    or getattr(action, "thought", None)
                    or getattr(action, "final_thought", None)
                    or getattr(action, "content", None)
                )

            try:
                action_payload = event_to_dict(action)
                action_args = action_payload.get("args")
            except Exception:
                action_args = None

            step_rec = {
                "iter": step_idx + 1,
                "action_type": type(action).__name__,
                "action": action_summary,
                "action_message": getattr(action, "message", None),
                "action_thought": getattr(action, "thought", None),
                "action_args": action_args,
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
                    try:
                        rec = event_to_dict(ev)
                    except Exception:
                        rec = {
                            "type": type(ev).__name__,
                            "source": getattr(ev, "source", None),
                            "content": getattr(ev, "content", None),
                            "command": getattr(ev, "command", None),
                            "thought": getattr(ev, "thought", None),
                            "outputs": getattr(ev, "outputs", None),
                            "path": getattr(ev, "path", None),
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
