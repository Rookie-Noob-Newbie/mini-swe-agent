import asyncio
import base64
import hashlib
import json
import os
import posixpath
import shlex
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

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
from openhands.events.action.commands import CmdRunAction
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


class _MiniRuntime(Runtime):
    """Minimal runtime that forwards CmdRunAction to mini-swe-agent Environment."""

    def __init__(self, env: Environment, *args, **kwargs):
        self.env = env
        self._undo_backups: dict[str, str] = {}
        self._undo_created: set[str] = set()
        super().__init__(*args, **kwargs)

    def _display_path(self, path: str) -> str:
        if path.startswith("/"):
            return path
        clean = path.lstrip("./")
        if not clean:
            return "/workspace"
        return f"/workspace/{clean}"

    def _path_exists(self, path: str) -> bool:
        resp = self.env.execute(f"test -e {shlex.quote(path)}")
        return resp.get("returncode", resp.get("exit_code", 0)) == 0

    def _resolve_path(self, path: str) -> str:
        if path.startswith("/workspace"):
            mapped = "/testbed" + path[len("/workspace") :]
        elif path.startswith("/testbed"):
            mapped = path
        elif path.startswith("/"):
            mapped = path
        else:
            mapped = f"/testbed/{path.lstrip('./')}"
        normalized = posixpath.normpath(mapped)
        if normalized == "/testbed":
            return normalized
        if not normalized.startswith("/testbed/"):
            raise ValueError("path is outside the workspace")
        return normalized

    def _run_python(self, script: str, env_vars: dict[str, str]) -> dict[str, Any]:
        env_prefix = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in env_vars.items()
        )
        cmd = f"{env_prefix} python - <<'PY'\n{script}\nPY"
        return self.env.execute(cmd)

    def _run_python_json(self, script: str, env_vars: dict[str, str]) -> dict[str, Any]:
        resp = self._run_python(script, env_vars)
        rc = resp.get("returncode", resp.get("exit_code", 0))
        output = resp.get("output", "").strip()
        if rc != 0:
            return {"ok": False, "error": output}
        if not output:
            return {"ok": False, "error": "empty output"}
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"ok": False, "error": output}

    def _stat_path(self, path: str) -> dict[str, Any]:
        script = """import json, os
path = os.environ["MSWEA_PATH"]
data = {
    "exists": os.path.exists(path),
    "is_dir": os.path.isdir(path),
    "is_file": os.path.isfile(path),
}
print(json.dumps(data))
"""
        data = self._run_python_json(script, {"MSWEA_PATH": path})
        if "exists" not in data:
            return {"exists": False, "is_dir": False, "is_file": False, "error": data.get("error", "")}
        return data

    def _read_file_text(self, path: str) -> tuple[str | None, str | None]:
        script = """import base64, json, os
path = os.environ["MSWEA_PATH"]
try:
    with open(path, "rb") as f:
        payload = base64.b64encode(f.read()).decode("ascii")
    print(json.dumps({"ok": True, "content_b64": payload}))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}))
"""
        data = self._run_python_json(script, {"MSWEA_PATH": path})
        if not data.get("ok"):
            return None, data.get("error", "file read failed")
        try:
            raw = base64.b64decode(data.get("content_b64", ""))
        except Exception as e:
            return None, str(e)
        return raw.decode("utf-8", errors="replace"), None

    def _write_file_text(self, path: str, content: str) -> str | None:
        payload = base64.b64encode(content.encode("utf-8")).decode("ascii")
        script = """import base64, json, os
path = os.environ["MSWEA_PATH"]
payload = base64.b64decode(os.environ["MSWEA_CONTENT_B64"])
try:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)
    print(json.dumps({"ok": True}))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}))
"""
        data = self._run_python_json(
            script,
            {"MSWEA_PATH": path, "MSWEA_CONTENT_B64": payload},
        )
        if not data.get("ok"):
            return data.get("error", "file write failed")
        return None

    def _list_directory(self, path: str) -> tuple[list[str], int, str | None]:
        script = """import json, os
root = os.environ["MSWEA_PATH"]
if not os.path.isdir(root):
    print(json.dumps({"ok": False, "error": "not a directory"}))
    raise SystemExit(0)
items = []
hidden_count = 0
for entry in sorted(os.listdir(root)):
    if entry.startswith("."):
        hidden_count += 1
        continue
    full = os.path.join(root, entry)
    if os.path.isdir(full):
        items.append(full + "/")
        for sub in sorted(os.listdir(full)):
            if sub.startswith("."):
                continue
            sub_full = os.path.join(full, sub)
            if os.path.isdir(sub_full):
                items.append(sub_full + "/")
            else:
                items.append(sub_full)
    else:
        items.append(full)
print(json.dumps({"ok": True, "items": items, "hidden_count": hidden_count}))
"""
        data = self._run_python_json(script, {"MSWEA_PATH": path})
        if not data.get("ok"):
            return [], 0, data.get("error", "list directory failed")
        return data.get("items", []), int(data.get("hidden_count", 0)), None

    def _ensure_lines(self, content: str) -> list[str]:
        if content == "":
            return [""]
        return content.splitlines()

    def _format_cat_n(self, lines: list[str], start_line: int = 1) -> str:
        if not lines:
            return ""
        width = max(6, len(str(start_line + len(lines) - 1)))
        return "\n".join(
            f"{line_no:>{width}}\t{line}"
            for line_no, line in enumerate(lines, start=start_line)
        )

    def _truncate_lines(self, lines: list[str], max_lines: int) -> tuple[list[str], bool]:
        if len(lines) <= max_lines:
            return lines, False
        return lines[:max_lines], True

    def _render_view_file(
        self, display_path: str, content: str, view_range: list[int] | None
    ) -> str:
        lines = self._ensure_lines(content)
        start_line = 1
        if view_range:
            try:
                start_line = max(int(view_range[0]), 1)
                end_line = int(view_range[1])
            except (TypeError, ValueError, IndexError):
                return "ERROR:\nInvalid `view_range` parameter."
            if end_line == -1 or end_line > len(lines):
                end_line = len(lines)
            if start_line > end_line:
                return "ERROR:\nInvalid `view_range` parameter."
            lines = lines[start_line - 1 : end_line]

        lines, truncated = self._truncate_lines(lines, max_lines=10000)
        cat_output = self._format_cat_n(lines, start_line=start_line)
        result = (
            f"Here's the result of running `cat -n` on {display_path}:\n{cat_output}"
        )
        if truncated:
            result += (
                "\nDue to the max output limit, only part of this file has been shown to you."
            )
        return result

    def _map_display_path(self, actual_path: str, actual_root: str, display_root: str) -> str:
        if actual_path.startswith(actual_root):
            suffix = actual_path[len(actual_root) :]
            if suffix.startswith("/"):
                return display_root + suffix
            if suffix:
                return f"{display_root}/{suffix}"
            return display_root
        return actual_path

    def _render_view_dir(self, actual_path: str, display_path: str) -> str:
        items, hidden_count, err = self._list_directory(actual_path)
        if err:
            return (
                f"ERROR:\nInvalid `path` parameter. The path {display_path} does not exist."
            )
        display_root = display_path.rstrip("/") or display_path
        root_line = display_root.rstrip("/") + "/"
        actual_root = actual_path.rstrip("/") or actual_path
        mapped = [
            self._map_display_path(item, actual_root, display_root) for item in items
        ]
        listing = "\n".join([root_line] + mapped)
        return (
            "Here's the files and directories up to 2 levels deep in "
            f"{display_root}, excluding hidden items:\n{listing}\n\n"
            f"{hidden_count} hidden files/directories in this directory are excluded. "
            "You can use 'ls -la /workspace' to see them."
        )

    def _render_view(
        self, actual_path: str, display_path: str, view_range: list[int] | None
    ) -> str:
        stat = self._stat_path(actual_path)
        if not stat.get("exists"):
            return (
                f"ERROR:\nInvalid `path` parameter. The path {display_path} does not exist."
            )
        if stat.get("is_dir"):
            return self._render_view_dir(actual_path, display_path)
        if not stat.get("is_file"):
            return (
                f"ERROR:\nInvalid `path` parameter. The path {display_path} does not exist."
            )
        content, err = self._read_file_text(actual_path)
        if err is not None:
            return f"ERROR:\n{err}"
        return self._render_view_file(display_path, content or "", view_range)

    def _slice_snippet(
        self,
        lines: list[str],
        start_line: int | None,
        end_line: int | None,
        context: int = 2,
        max_lines: int = 20,
    ) -> tuple[list[str], int, bool]:
        if not lines:
            return [""], 1, False
        if start_line is None or end_line is None:
            snippet_start = 1
            snippet_end = min(len(lines), max_lines)
        else:
            snippet_start = max(start_line - context, 1)
            snippet_end = min(end_line + context, len(lines))
            if snippet_end - snippet_start + 1 > max_lines:
                snippet_start = max(start_line - 1, 1)
                snippet_end = min(snippet_start + max_lines - 1, len(lines))
        snippet_lines = lines[snippet_start - 1 : snippet_end]
        truncated = len(snippet_lines) < len(lines)
        return snippet_lines, snippet_start, truncated

    def _render_edit_result(
        self, display_path: str, content: str, start_line: int | None, end_line: int | None
    ) -> str:
        lines = self._ensure_lines(content)
        snippet_lines, snippet_start, truncated = self._slice_snippet(
            lines, start_line, end_line
        )
        cat_output = self._format_cat_n(snippet_lines, start_line=snippet_start)
        result = (
            f"The file {display_path} has been edited. Here's the result of running "
            f"`cat -n` on a snippet of {display_path}:\n{cat_output}\n"
            "Review the changes and make sure they are as expected. Edit the file again if necessary."
        )
        if truncated:
            result += (
                "\nDue to the max output limit, only part of this file has been shown to you."
            )
        return result

    def _render_undo_result(self, display_path: str, content: str) -> str:
        lines = self._ensure_lines(content)
        snippet_lines, snippet_start, truncated = self._slice_snippet(
            lines, 1, min(len(lines), 1)
        )
        cat_output = self._format_cat_n(snippet_lines, start_line=snippet_start)
        result = (
            f"Last edit to {display_path} was undone successfully. Here's the result "
            f"of running `cat -n` on a snippet of {display_path}:\n{cat_output}\n"
            "Review the changes and make sure they are as expected. Edit the file again if necessary."
        )
        if truncated:
            result += (
                "\nDue to the max output limit, only part of this file has been shown to you."
            )
        return result

    def _find_occurrences(self, content: str, old_str: str) -> tuple[list[int], int]:
        if old_str == "":
            lines = self._ensure_lines(content)
            return list(range(1, len(lines) + 1)), len(lines)
        indices: list[int] = []
        start = 0
        while True:
            idx = content.find(old_str, start)
            if idx == -1:
                break
            indices.append(idx)
            start = idx + 1
        line_numbers: list[int] = []
        for idx in indices:
            line_no = content.count("\n", 0, idx) + 1
            if line_no not in line_numbers:
                line_numbers.append(line_no)
        return line_numbers, len(indices)

    def _read_file_numbered(self, path: str, start: int, end: int) -> dict[str, Any]:
        script = """import os, sys
path = os.environ["MSWEA_PATH"]
start = int(os.environ.get("MSWEA_START", "1"))
end = int(os.environ.get("MSWEA_END", "-1"))
with open(path, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()
if end == -1 or end > len(lines):
    end = len(lines)
start_idx = max(start - 1, 0)
end_idx = max(end, 0)
for idx in range(start_idx, end_idx):
    sys.stdout.write(f"{idx + 1}\\t{lines[idx]}")
"""
        return self._run_python(
            script,
            {
                "MSWEA_PATH": path,
                "MSWEA_START": str(start),
                "MSWEA_END": str(end),
            },
        )

    def _read_file_raw(self, path: str) -> dict[str, Any]:
        script = """import os, sys
path = os.environ["MSWEA_PATH"]
with open(path, "r", encoding="utf-8", errors="replace") as f:
    sys.stdout.write(f.read())
"""
        return self._run_python(script, {"MSWEA_PATH": path})

    def _write_file_raw(self, path: str, content: str) -> dict[str, Any]:
        payload = base64.b64encode(content.encode("utf-8")).decode("ascii")
        script = """import base64, os
path = os.environ["MSWEA_PATH"]
payload = base64.b64decode(os.environ["MSWEA_CONTENT_B64"]).decode("utf-8")
parent = os.path.dirname(path)
if parent:
    os.makedirs(parent, exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    f.write(payload)
"""
        return self._run_python(
            script,
            {
                "MSWEA_PATH": path,
                "MSWEA_CONTENT_B64": payload,
            },
        )

    def _backup_path(self, path: str) -> str:
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
        return f"/tmp/openhands_undo/{digest}.bak"

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
        if action.impl_source == FileReadSource.OH_ACI:
            display_path = self._display_path(action.path)
            try:
                actual_path = self._resolve_path(action.path)
            except ValueError:
                content = (
                    f"ERROR:\nInvalid `path` parameter. The path {display_path} does not exist."
                )
                return FileReadObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileReadSource.OH_ACI,
                )
            content = self._render_view(actual_path, display_path, action.view_range)
            return FileReadObservation(
                content=content,
                path=display_path,
                impl_source=FileReadSource.OH_ACI,
            )

        try:
            path = self._resolve_path(action.path)
        except ValueError as e:
            return ErrorObservation(f"Invalid path {action.path}: {e}")
        resp = self._read_file_raw(path)
        rc = resp.get("returncode", resp.get("exit_code", 0))
        if rc != 0:
            return ErrorObservation(
                f"File read failed for {action.path}: {resp.get('output', '')}"
            )
        return FileReadObservation(content=resp.get("output", ""), path=action.path)

    def write(self, action) -> Observation:
        try:
            path = self._resolve_path(action.path)
        except ValueError as e:
            return ErrorObservation(f"Invalid path {action.path}: {e}")
        resp = self._write_file_raw(path, action.content)
        rc = resp.get("returncode", resp.get("exit_code", 0))
        if rc != 0:
            return ErrorObservation(
                f"File write failed for {action.path}: {resp.get('output', '')}"
            )
        return FileWriteObservation(content="", path=action.path)

    def edit(self, action) -> Observation:
        if action.impl_source != FileEditSource.OH_ACI:
            return ErrorObservation(
                "Only OH_ACI file edits are supported in mini runtime."
            )
        display_path = self._display_path(action.path)
        try:
            actual_path = self._resolve_path(action.path)
        except ValueError:
            content = (
                f"ERROR:\nInvalid `path` parameter. The path {display_path} does not exist."
            )
            return FileEditObservation(
                content=content,
                path=display_path,
                impl_source=FileEditSource.OH_ACI,
            )

        if action.command == "view":
            content = self._render_view(actual_path, display_path, None)
            return FileEditObservation(
                content=content,
                path=display_path,
                impl_source=FileEditSource.OH_ACI,
            )

        if action.command == "create":
            if action.file_text is None:
                content = "ERROR:\nParameter `file_text` is required for command: create."
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            stat = self._stat_path(actual_path)
            if stat.get("exists"):
                content = f"ERROR:\nFile already exists at: {display_path}"
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            parent = posixpath.dirname(actual_path)
            if parent and not self._stat_path(parent).get("exists"):
                content = (
                    f"ERROR:\nInvalid `path` parameter. The path {display_path} does not exist."
                )
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            err = self._write_file_text(actual_path, action.file_text or "")
            if err is not None:
                content = f"ERROR:\n{err}"
            else:
                self._undo_created.add(actual_path)
                self._undo_backups.pop(actual_path, None)
                content = f"File created successfully at: {display_path}"
            return FileEditObservation(
                content=content,
                path=display_path,
                impl_source=FileEditSource.OH_ACI,
            )

        if action.command == "str_replace":
            if action.old_str is None:
                content = "ERROR:\nParameter `old_str` is required for command: str_replace."
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            new_str = action.new_str if action.new_str is not None else ""
            if action.old_str == new_str:
                content = (
                    "No replacement was performed. `new_str` and `old_str` must be different."
                )
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            stat = self._stat_path(actual_path)
            if not stat.get("exists") or stat.get("is_dir"):
                content = (
                    f"ERROR:\nInvalid `path` parameter. The path {display_path} does not exist."
                )
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            content_text, err = self._read_file_text(actual_path)
            if err is not None:
                content = f"ERROR:\n{err}"
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            content_text = content_text or ""
            line_numbers, occurrence_count = self._find_occurrences(
                content_text, action.old_str
            )
            if occurrence_count == 0:
                content = (
                    "No replacement was performed. "
                    f"old_str `{action.old_str}` did not appear verbatim in {display_path}."
                )
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            if occurrence_count > 1:
                line_list = "[" + ", ".join(str(n) for n in line_numbers) + "]"
                content = (
                    "No replacement was performed. Multiple occurrences of old_str "
                    f"`{action.old_str}` in lines {line_list}. Please ensure it is unique."
                )
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )

            self._undo_created.discard(actual_path)
            backup_path = self._backup_path(actual_path)
            backup_err = self._write_file_text(backup_path, content_text)
            if backup_err is None:
                self._undo_backups[actual_path] = backup_path

            new_content = content_text.replace(action.old_str, new_str, 1)
            write_err = self._write_file_text(actual_path, new_content)
            if write_err is not None:
                content = f"ERROR:\n{write_err}"
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )

            start_line = line_numbers[0] if line_numbers else 1
            old_line_count = max(1, action.old_str.count("\n") + 1)
            new_line_count = max(1, new_str.count("\n") + 1)
            end_line = start_line + max(old_line_count, new_line_count) - 1
            content = self._render_edit_result(
                display_path, new_content, start_line, end_line
            )
            return FileEditObservation(
                content=content,
                path=display_path,
                impl_source=FileEditSource.OH_ACI,
            )

        if action.command == "insert":
            if action.insert_line is None:
                content = "ERROR:\nParameter `insert_line` is required for command: insert."
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            if action.new_str is None:
                content = "ERROR:\nParameter `new_str` is required for command: insert."
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            insert_line = action.insert_line
            if isinstance(insert_line, str):
                try:
                    insert_line = int(insert_line)
                except ValueError:
                    content = (
                        f"ERROR:\nInvalid insert_line value: '{insert_line}'. Expected an integer."
                    )
                    return FileEditObservation(
                        content=content,
                        path=display_path,
                        impl_source=FileEditSource.OH_ACI,
                    )
            stat = self._stat_path(actual_path)
            if not stat.get("exists") or stat.get("is_dir"):
                content = (
                    f"ERROR:\nInvalid `path` parameter. The path {display_path} does not exist."
                )
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            content_text, err = self._read_file_text(actual_path)
            if err is not None:
                content = f"ERROR:\n{err}"
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            content_text = content_text or ""
            lines = self._ensure_lines(content_text)
            max_line = len(lines)
            if insert_line < 0 or insert_line > max_line:
                content = (
                    "ERROR:\nInvalid `insert_line` parameter. It should be within "
                    f"the range of allowed values [0, {max_line}]."
                )
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            new_lines = action.new_str.splitlines()
            if action.new_str == "":
                new_lines = [""]
            insert_idx = insert_line
            lines[insert_idx:insert_idx] = new_lines

            self._undo_created.discard(actual_path)
            backup_path = self._backup_path(actual_path)
            backup_err = self._write_file_text(backup_path, content_text)
            if backup_err is None:
                self._undo_backups[actual_path] = backup_path

            new_content = "\n".join(lines)
            write_err = self._write_file_text(actual_path, new_content)
            if write_err is not None:
                content = f"ERROR:\n{write_err}"
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )

            start_line = insert_line + 1
            end_line = start_line + max(1, len(new_lines)) - 1
            content = self._render_edit_result(
                display_path, new_content, start_line, end_line
            )
            return FileEditObservation(
                content=content,
                path=display_path,
                impl_source=FileEditSource.OH_ACI,
            )

        if action.command == "undo_edit":
            if actual_path in self._undo_created:
                resp = self.env.execute(f"rm -f {shlex.quote(actual_path)}")
                rc = resp.get("returncode", resp.get("exit_code", 0))
                if rc != 0:
                    content = f"ERROR:\nUndo edit failed for {display_path}: {resp.get('output', '')}"
                else:
                    self._undo_created.discard(actual_path)
                    content = f"Last edit to {display_path} was undone successfully."
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            if actual_path not in self._undo_backups:
                content = f"ERROR:\nNo edit history found for {display_path}."
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            resp = self.env.execute(
                f"cp {shlex.quote(self._undo_backups[actual_path])} {shlex.quote(actual_path)}"
            )
            rc = resp.get("returncode", resp.get("exit_code", 0))
            if rc != 0:
                content = f"ERROR:\nUndo edit failed for {display_path}: {resp.get('output', '')}"
                return FileEditObservation(
                    content=content,
                    path=display_path,
                    impl_source=FileEditSource.OH_ACI,
                )
            self._undo_backups.pop(actual_path, None)
            content_text, err = self._read_file_text(actual_path)
            if err is not None:
                content = f"ERROR:\n{err}"
            else:
                content = self._render_undo_result(display_path, content_text or "")
            return FileEditObservation(
                content=content,
                path=display_path,
                impl_source=FileEditSource.OH_ACI,
            )

        content = f"ERROR:\nUnsupported file edit command '{action.command}'."
        return FileEditObservation(
            content=content,
            path=display_path,
            impl_source=FileEditSource.OH_ACI,
        )

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
            enable_browsing=False,
            enable_jupyter=False,
            enable_editor=True,
            enable_llm_editor=False,
            enable_plan_mode=True,
            enable_condensation_request=False,
            enable_prompt_extensions=False,
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
