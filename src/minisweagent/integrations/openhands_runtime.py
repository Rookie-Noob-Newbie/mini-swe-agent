from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import posixpath
import re
import shlex
import subprocess
import threading
import time
import uuid
from typing import Any

from minisweagent import Environment
from minisweagent.utils.log import logger as ms_logger

from openhands.core.config import MCPConfig
from openhands.events import EventStreamSubscriber
from openhands.events.action import FileEditAction, FileReadAction, FileWriteAction
from openhands.events.action.browse import BrowseInteractiveAction, BrowseURLAction
from openhands.events.action.commands import CmdRunAction, IPythonRunCellAction
from openhands.events.action.mcp import MCPAction
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.observation import (
    BrowserOutputObservation,
    CmdOutputObservation,
    ErrorObservation,
    FileEditObservation,
    FileReadObservation,
    FileWriteObservation,
    IPythonRunCellObservation,
    MCPObservation,
    Observation,
)
from openhands.runtime.base import Runtime

_TIMEOUT_MESSAGE_TEMPLATE = (
    "You may wait longer to see additional output by sending empty command '', "
    'send other commands to interact with the current process, '
    'send keys ("C-c", "C-z", "C-d") to interrupt/kill the previous command before sending your new command, '
    "or use the timeout parameter in execute_bash for future commands."
)


class _DockerExecSession:
    def __init__(
        self,
        *,
        docker_executable: str,
        container_id: str,
        base_cwd: str,
        env_vars: dict[str, str],
        forward_env: list[str],
        no_change_timeout: int = 30,
        poll_interval: float = 0.5,
    ) -> None:
        self.docker_executable = docker_executable
        self.container_id = container_id
        self.base_cwd = base_cwd or "/"
        self.env_vars = env_vars
        self.forward_env = forward_env
        self.no_change_timeout = no_change_timeout
        self.poll_interval = poll_interval
        self._proc: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._buffer = io.StringIO()
        self._buffer_lock = threading.Lock()
        self._last_output_time: float | None = None
        self._pending_start_marker: str | None = None
        self._pending_end_marker: str | None = None
        self._pending_output_start: int | None = None
        self._pending_output_cursor: int | None = None
        self._pending_start_time: float | None = None

    @classmethod
    def from_env(cls, env: Environment) -> "_DockerExecSession | None":
        container_id = getattr(env, "container_id", None)
        cfg = getattr(env, "config", None)
        docker_executable = getattr(cfg, "executable", None) if cfg is not None else None
        base_cwd = getattr(cfg, "cwd", "/") if cfg is not None else "/"
        env_vars = getattr(cfg, "env", {}) if cfg is not None else {}
        forward_env = getattr(cfg, "forward_env", []) if cfg is not None else []
        if not container_id or not docker_executable:
            return None
        return cls(
            docker_executable=str(docker_executable),
            container_id=str(container_id),
            base_cwd=str(base_cwd),
            env_vars=dict(env_vars or {}),
            forward_env=list(forward_env or []),
        )

    def _reset_buffer(self) -> None:
        with self._buffer_lock:
            self._buffer = io.StringIO()

    def _trim_buffer(self, upto: int) -> None:
        if upto <= 0:
            return
        with self._buffer_lock:
            data = self._buffer.getvalue()
            if upto >= len(data):
                self._buffer = io.StringIO()
                self._pending_output_start = None
                self._pending_output_cursor = None
                return
            remaining = data[upto:]
            self._buffer = io.StringIO()
            self._buffer.write(remaining)
            if self._pending_output_start is not None:
                self._pending_output_start = max(self._pending_output_start - upto, 0)
            if self._pending_output_cursor is not None:
                self._pending_output_cursor = max(self._pending_output_cursor - upto, 0)

    def _clear_pending(self) -> None:
        self._pending_start_marker = None
        self._pending_end_marker = None
        self._pending_output_start = None
        self._pending_output_cursor = None
        self._pending_start_time = None

    def _reader(self, stream: io.TextIOBase) -> None:
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                with self._buffer_lock:
                    self._buffer.write(chunk)
                    self._last_output_time = time.time()
        except Exception:
            pass

    def _build_exec_cmd(self) -> list[str]:
        cmd = [self.docker_executable, "exec", "-i", "-w", self.base_cwd]
        for key in self.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.env_vars.items():
            cmd.extend(["-e", f"{key}={str(value)}"])
        cmd.extend([self.container_id, "bash", "-i"])
        return cmd

    def _start_shell(self) -> None:
        self._reset_buffer()
        self._clear_pending()
        exec_cmd = self._build_exec_cmd()
        self._proc = subprocess.Popen(
            exec_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._last_output_time = time.time()
        if self._proc.stdout is not None:
            self._reader_thread = threading.Thread(
                target=self._reader,
                args=(self._proc.stdout,),
                daemon=True,
            )
            self._reader_thread.start()
        self._write_raw("export PS1=''\n")

    def _ensure_shell(self) -> bool:
        if self._proc is None or self._proc.poll() is not None:
            self._start_shell()
        return self._proc is not None

    def _write_raw(self, payload: str) -> None:
        if not self._proc or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()
        except Exception:
            return

    def _send_input(self, command: str) -> None:
        if command in {"C-c", "C-z", "C-d"}:
            mapping = {"C-c": "\x03", "C-z": "\x1a", "C-d": "\x04"}
            self._write_raw(mapping[command])
            return
        if command:
            self._write_raw(command + "\n")

    def _wrap_command(self, command: str, cwd: str, start_marker: str, end_marker: str) -> str:
        command = command.replace("\r\n", "\n").replace("\r", "\n")
        if cwd:
            command = f"cd {shlex.quote(cwd)}\n{command}"
        return (
            f'printf "%s\\n" "{start_marker}"\n'
            f'{{\n{command}\n}};\n'
            "__MSWEA_RC=$?\n"
            f'printf "%s%d\\n" "{end_marker}" "${__MSWEA_RC}"\n'
        )

    def _start_command(self, command: str, cwd: str) -> None:
        marker = uuid.uuid4().hex[:8]
        start_marker = f"__MSWEA_START__{marker}__"
        end_marker = f"__MSWEA_END__{marker}__RC__"
        self._pending_start_marker = start_marker
        self._pending_end_marker = end_marker
        self._pending_output_start = None
        self._pending_output_cursor = None
        self._pending_start_time = time.time()
        self._reset_buffer()
        wrapped = self._wrap_command(command, cwd, start_marker, end_marker)
        self._write_raw(wrapped + "\n")

    def _collect_pending_output(self) -> tuple[str, bool, int | None]:
        if not self._pending_start_marker or not self._pending_end_marker:
            return "", False, None
        with self._buffer_lock:
            data = self._buffer.getvalue()
        if self._pending_output_start is None:
            start_idx = data.find(self._pending_start_marker)
            if start_idx != -1:
                start_idx += len(self._pending_start_marker)
                if start_idx < len(data) and data[start_idx] == "\n":
                    start_idx += 1
                self._pending_output_start = start_idx
                self._pending_output_cursor = start_idx
        if self._pending_output_start is None:
            return "", False, None
        end_idx = data.find(self._pending_end_marker, self._pending_output_start)
        if end_idx != -1:
            rc_start = end_idx + len(self._pending_end_marker)
            rc_end = data.find("\n", rc_start)
            if rc_end == -1:
                rc_end = len(data)
            rc_str = data[rc_start:rc_end].strip()
            try:
                exit_code = int(rc_str)
            except ValueError:
                exit_code = 0
            cursor = self._pending_output_cursor or self._pending_output_start
            output = data[cursor:end_idx]
            trim_to = rc_end + 1 if rc_end < len(data) else rc_end
            self._trim_buffer(trim_to)
            self._clear_pending()
            return output, True, exit_code
        cursor = self._pending_output_cursor or self._pending_output_start
        output = data[cursor:]
        if self._pending_output_cursor is not None:
            self._pending_output_cursor = len(data)
        return output, False, None

    def _wait_for_pending(
        self,
        *,
        hard_timeout: float | None,
        allow_no_change: bool,
    ) -> tuple[str, int, str]:
        output_parts: list[str] = []
        start_time = self._pending_start_time or time.time()
        while True:
            chunk, finished, exit_code = self._collect_pending_output()
            if chunk:
                output_parts.append(chunk)
            if finished:
                return "".join(output_parts), exit_code or 0, "finished"
            now = time.time()
            if hard_timeout is not None and now - start_time >= hard_timeout:
                return "".join(output_parts), -1, "timeout"
            last_output = self._last_output_time or start_time
            if allow_no_change and now - last_output >= self.no_change_timeout:
                return "".join(output_parts), -1, "no_change"
            time.sleep(self.poll_interval)

    def run(
        self,
        *,
        command: str,
        cwd: str,
        hard_timeout: float | None,
        is_input: bool,
        blocking: bool,
    ) -> tuple[str, int, str]:
        if not self._ensure_shell():
            return "ERROR: Failed to start shell.", -1, "error"
        has_pending = self._pending_start_marker is not None
        if has_pending:
            if not is_input and command != "":
                output, _, _ = self._collect_pending_output()
                if output:
                    output = "[Below is the output of the previous command.]\n" + output
                output += (
                    f'\n[Your command "{command}" is NOT executed. '
                    "The previous command is still running - You CANNOT send new commands until the previous command is completed. "
                    f"{_TIMEOUT_MESSAGE_TEMPLATE}]"
                )
                return output, -1, "busy"
            if command:
                self._send_input(command)
            return self._wait_for_pending(
                hard_timeout=hard_timeout,
                allow_no_change=not blocking,
            )
        if command == "":
            return "ERROR: No previous running command to retrieve logs from.", -1, "no_prev"
        if is_input:
            return "ERROR: No previous running command to interact with.", -1, "no_prev"
        self._start_command(command, cwd)
        return self._wait_for_pending(
            hard_timeout=hard_timeout,
            allow_no_change=not blocking,
        )

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._write_raw("exit\n")
        except Exception:
            pass
        try:
            self._proc.terminate()
        except Exception:
            pass
        try:
            self._proc.kill()
        except Exception:
            pass
        self._proc = None
        self._reader_thread = None
        self._reset_buffer()


class OpenHandsCompatRuntime(Runtime):
    """Runtime adapter that mirrors OpenHands tool behavior via mini-swe-agent."""

    def __init__(self, env: Environment, *args, **kwargs):
        self.env = env
        self._cmd_session = _DockerExecSession.from_env(env)
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

    def _display_from_actual(self, path: str) -> str:
        if path.startswith("/testbed"):
            suffix = path[len("/testbed") :]
            if suffix.startswith("/"):
                return "/workspace" + suffix
            return "/workspace"
        return path

    def _path_exists(self, path: str) -> bool:
        resp = self.env.execute(f"test -e {shlex.quote(path)}")
        return resp.get("returncode", resp.get("exit_code", 0)) == 0

    def _resolve_path(self, path: str) -> str:
        if path.startswith("/workspace"):
            mapped = path
        elif path.startswith("/testbed"):
            mapped = path
        elif path.startswith("/"):
            mapped = path
        else:
            base = "/workspace"
            if not self._path_exists(base) and self._path_exists("/testbed"):
                base = "/testbed"
            mapped = f"{base}/{path.lstrip('./')}"
        normalized = posixpath.normpath(mapped)
        if normalized in {"/workspace", "/testbed"}:
            return normalized
        if normalized.startswith("/workspace/") or normalized.startswith("/testbed/"):
            return normalized
        raise ValueError("path is outside the workspace")

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
            return {
                "exists": False,
                "is_dir": False,
                "is_file": False,
                "error": data.get("error", ""),
            }
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
        payload_bytes = content.encode("utf-8")
        payload = base64.b64encode(payload_bytes).decode("ascii")
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
        max_payload = 60000
        if len(payload) <= max_payload:
            data = self._run_python_json(
                script,
                {"MSWEA_PATH": path, "MSWEA_CONTENT_B64": payload},
            )
            if not data.get("ok"):
                return data.get("error", "file write failed")
            return None

        chunk_script = """import base64, os
path = os.environ["MSWEA_PATH"]
payload = base64.b64decode(os.environ["MSWEA_CONTENT_B64"])
parent = os.path.dirname(path)
if parent:
    os.makedirs(parent, exist_ok=True)
mode = "ab" if os.environ.get("MSWEA_APPEND") == "1" else "wb"
with open(path, mode) as f:
    f.write(payload)
"""
        chunk_size = max(1, (max_payload // 4) * 3)
        offset = 0
        while offset < len(payload_bytes):
            chunk = payload_bytes[offset : offset + chunk_size]
            chunk_b64 = base64.b64encode(chunk).decode("ascii")
            resp = self._run_python(
                chunk_script,
                {
                    "MSWEA_PATH": path,
                    "MSWEA_CONTENT_B64": chunk_b64,
                    "MSWEA_APPEND": "1" if offset else "0",
                },
            )
            rc = resp.get("returncode", resp.get("exit_code", 0))
            if rc != 0:
                return resp.get("output", "file write failed")
            offset += len(chunk)
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

    def _map_display_path(
        self, actual_path: str, actual_root: str, display_root: str
    ) -> str:
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
        self,
        display_path: str,
        content: str,
        start_line: int | None,
        end_line: int | None,
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

    def _backup_path(self, path: str) -> str:
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
        return f"/tmp/openhands_undo/{digest}.bak"

    def _get_diff(self, old: str, new: str, path: str) -> str:
        try:
            from openhands_aci.utils.diff import get_diff as oh_get_diff

            return oh_get_diff(old_contents=old, new_contents=new, filepath=path)
        except Exception:
            import difflib

            return "\n".join(
                difflib.unified_diff(
                    old.splitlines(),
                    new.splitlines(),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                    lineterm="",
                )
            )
    async def connect(self) -> None:  # pragma: no cover - no-op for this adapter
        self._runtime_initialized = True

    def close(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.event_stream.unsubscribe(EventStreamSubscriber.RUNTIME, self.sid)
        except Exception:
            pass
        if self._cmd_session is not None:
            try:
                self._cmd_session.close()
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
        if self._cmd_session is not None:
            try:
                output, exit_code, status = self._cmd_session.run(
                    command=action.command,
                    cwd=cwd,
                    hard_timeout=action.timeout,
                    is_input=action.is_input,
                    blocking=action.blocking,
                )
                if status == "no_change":
                    suffix = (
                        f"[The command has no new output after {self._cmd_session.no_change_timeout} seconds. "
                        f"{_TIMEOUT_MESSAGE_TEMPLATE}]"
                    )
                    if output:
                        output = "[Below is the output of the previous command.]\n" + output
                    output = (output + "\n" if output else "") + suffix
                elif status == "timeout":
                    suffix = (
                        f"[The command timed out after {action.timeout} seconds. {_TIMEOUT_MESSAGE_TEMPLATE}]"
                    )
                    if output:
                        output = "[Below is the output of the previous command.]\n" + output
                    output = (output + "\n" if output else "") + suffix
                display_cwd = self._display_from_actual(cwd) if cwd else "/workspace"
                metadata = {"exit_code": exit_code, "working_dir": display_cwd}
                return CmdOutputObservation(
                    content=output,
                    command=action.command,
                    metadata=metadata,
                )
            except Exception as e:  # pragma: no cover
                ms_logger.error(f"[OpenHandsCompatRuntime] interactive run failed: {e}")
                return ErrorObservation(f"Runtime execution failed: {e}")
        if action.is_input:
            return ErrorObservation(
                "CLIRuntime does not support interactive input from the agent."
            )
        try:
            result = self.env.execute(action.command, cwd=cwd)
            output = result.get("output", "")
            exit_code = result.get("returncode", result.get("exit_code", 0))
            display_cwd = self._display_from_actual(cwd) if cwd else "/workspace"
            metadata = {"exit_code": exit_code, "working_dir": display_cwd}
            return CmdOutputObservation(
                content=output,
                command=action.command,
                metadata=metadata,
            )
        except Exception as e:  # pragma: no cover
            return ErrorObservation(f"Runtime execution failed: {e}")

    def _translate_ipython_magics(self, code: str) -> str:
        lines = code.splitlines()
        processed: list[str] = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("%pip "):
                cmd = stripped[len("%pip ") :].strip()
                processed.append("import subprocess as _mswea_subprocess")
                processed.append(
                    f"_mswea_proc = _mswea_subprocess.run({json.dumps('pip ' + cmd)}, shell=True, capture_output=True, text=True)"
                )
                processed.append(
                    "print(_mswea_proc.stdout + _mswea_proc.stderr, end='')"
                )
            elif stripped.startswith("!"):
                cmd = stripped[1:].strip()
                processed.append("import subprocess as _mswea_subprocess")
                processed.append(
                    f"_mswea_proc = _mswea_subprocess.run({json.dumps(cmd)}, shell=True, capture_output=True, text=True)"
                )
                processed.append(
                    "print(_mswea_proc.stdout + _mswea_proc.stderr, end='')"
                )
            else:
                processed.append(line)
        return "\n".join(processed)

    def _get_working_directory(self) -> str:
        resp = self.env.execute("pwd")
        output = (resp.get("output", "") or "").strip()
        if not output:
            return "/workspace"
        return self._display_from_actual(output)

    def _get_python_path(self) -> str:
        resp = self.env.execute("which python")
        output = (resp.get("output", "") or "").strip()
        return output or "python"

    def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        code = self._translate_ipython_magics(action.code)
        payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
        script = """import base64, traceback
code = base64.b64decode(os.environ["MSWEA_CODE_B64"]).decode("utf-8")
try:
    exec(compile(code, "<ipython>", "exec"), globals(), globals())
except Exception:
    traceback.print_exc()
"""
        resp = self._run_python(script, {"MSWEA_CODE_B64": payload})
        output = resp.get("output", "")
        if not output.strip():
            output = "[Code executed successfully with no output]"
        if action.include_extra:
            output += f"\n[Jupyter current working directory: {self._get_working_directory()}]"
            output += f"\n[Jupyter Python interpreter: {self._get_python_path()}]"
        return IPythonRunCellObservation(content=output, code=action.code)
    def read(self, action: FileReadAction) -> Observation:
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
        resp = self._run_python(
            """import os, sys
path = os.environ["MSWEA_PATH"]
with open(path, "r", encoding="utf-8", errors="replace") as f:
    sys.stdout.write(f.read())
""",
            {"MSWEA_PATH": path},
        )
        rc = resp.get("returncode", resp.get("exit_code", 0))
        if rc != 0:
            return ErrorObservation(
                f"File read failed for {action.path}: {resp.get('output', '')}"
            )
        return FileReadObservation(content=resp.get("output", ""), path=action.path)

    def write(self, action: FileWriteAction) -> Observation:
        try:
            path = self._resolve_path(action.path)
        except ValueError as e:
            return ErrorObservation(f"Invalid path {action.path}: {e}")
        write_err = self._write_file_text(path, action.content)
        if write_err is not None:
            return ErrorObservation(f"File write failed for {action.path}: {write_err}")
        return FileWriteObservation(content="", path=action.path)

    def _validate_llm_range(self, start: int, end: int, total_lines: int) -> str | None:
        if (
            (start < 1 and start != -1)
            or start > total_lines
            or (start > end and end != -1 and start != -1)
        ):
            return (
                f"Invalid range for editing: start={start}, end={end}, total lines={total_lines}. "
                f"start must be >= 1 and <={total_lines} (total lines of the edited file), "
                "start <= end, or start == -1 (append to the end of the file)."
            )
        if (
            (end < 1 and end != -1)
            or end > total_lines
            or (end < start and start != -1 and end != -1)
        ):
            return (
                f"Invalid range for editing: start={start}, end={end}, total lines={total_lines}. "
                f"end must be >= 1 and <= {total_lines} (total lines of the edited file), "
                "end >= start, or end == -1 (to edit till the end of the file)."
            )
        return None

    def _edit_llm_based(self, action: FileEditAction) -> Observation:
        display_path = self._display_path(action.path)
        try:
            actual_path = self._resolve_path(action.path)
        except ValueError:
            return ErrorObservation(
                f"Invalid `path` parameter. The path {display_path} does not exist."
            )
        stat = self._stat_path(actual_path)
        if not stat.get("exists"):
            write_err = self._write_file_text(actual_path, action.content.strip())
            if write_err is not None:
                return ErrorObservation(write_err)
            diff = self._get_diff("", action.content, display_path)
            return FileEditObservation(
                content=diff,
                path=display_path,
                prev_exist=False,
                old_content="",
                new_content=action.content,
            )

        old_content, err = self._read_file_text(actual_path)
        if err is not None:
            return ErrorObservation(err)
        old_content = old_content or ""
        old_lines = old_content.split("\n")
        start = action.start
        end = action.end
        error = self._validate_llm_range(start, end, len(old_lines))
        if error is not None:
            return ErrorObservation(error)

        if start == -1:
            updated_content = "\n".join(old_lines + action.content.split("\n"))
        else:
            start_idx = start - 1
            if end != -1:
                end_idx = end
            else:
                end_idx = len(old_lines)
            updated_content = "\n".join(
                old_lines[:start_idx]
                + action.content.split("\n")
                + old_lines[end_idx:]
            )

        write_err = self._write_file_text(actual_path, updated_content)
        if write_err is not None:
            return ErrorObservation(write_err)
        diff = self._get_diff(old_content, updated_content, display_path)
        return FileEditObservation(
            content=diff,
            path=display_path,
            prev_exist=True,
            old_content=old_content,
            new_content=updated_content,
        )
    def edit(self, action: FileEditAction) -> Observation:
        if action.impl_source == FileEditSource.LLM_BASED_EDIT:
            return self._edit_llm_based(action)

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
                    content = (
                        f"ERROR:\nUndo edit failed for {display_path}: {resp.get('output', '')}"
                    )
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
                content = (
                    f"ERROR:\nUndo edit failed for {display_path}: {resp.get('output', '')}"
                )
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

    def browse(self, action: BrowseURLAction) -> Observation:
        return ErrorObservation("Browser functionality is not supported or disabled.")

    def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        return ErrorObservation("Browser functionality is not supported or disabled.")

    async def call_tool_mcp(self, action: MCPAction) -> Observation:
        import sys

        if sys.platform == "win32":
            return ErrorObservation("MCP functionality is not available on Windows")

        mcp_config = self.get_mcp_config()
        if (
            not mcp_config.sse_servers
            and not mcp_config.shttp_servers
            and not mcp_config.stdio_servers
        ):
            return ErrorObservation("No MCP servers configured")

        try:
            from openhands.mcp.utils import call_tool_mcp as call_tool_mcp_handler
            from openhands.mcp.utils import create_mcp_clients
        except Exception as e:
            return ErrorObservation(f"Error executing MCP tool {action.name}: {e}")

        try:
            mcp_clients = await create_mcp_clients(
                mcp_config.sse_servers,
                mcp_config.shttp_servers,
                self.sid,
                mcp_config.stdio_servers,
            )
            if not mcp_clients:
                return ErrorObservation(
                    "No MCP clients could be created - check server configurations"
                )
            return await call_tool_mcp_handler(mcp_clients, action)
        except Exception as e:
            return ErrorObservation(f"Error executing MCP tool {action.name}: {e}")

    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        if not os.path.exists(host_src):
            raise FileNotFoundError(f"Source path '{host_src}' does not exist.")

        dest_display = sandbox_dest or "/workspace"
        try:
            dest_actual = self._resolve_path(dest_display)
        except ValueError as e:
            raise FileNotFoundError(f"Invalid destination path '{sandbox_dest}': {e}")

        if os.path.isdir(host_src):
            if not recursive:
                raise FileNotFoundError(
                    f"Source path '{host_src}' is a directory; set recursive=True to copy."
                )
            target_dir = posixpath.join(dest_actual, os.path.basename(host_src))
            import io
            import zipfile

            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(host_src):
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        arcname = os.path.relpath(file_path, host_src)
                        zipf.write(file_path, arcname)
            payload = base64.b64encode(buffer.getvalue()).decode("ascii")
            script = """import base64, io, os, zipfile
path = os.environ["MSWEA_PATH"]
payload = base64.b64decode(os.environ["MSWEA_CONTENT_B64"])
os.makedirs(path, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as zipf:
    zipf.extractall(path)
"""
            resp = self._run_python(
                script,
                {
                    "MSWEA_PATH": target_dir,
                    "MSWEA_CONTENT_B64": payload,
                },
            )
            rc = resp.get("returncode", resp.get("exit_code", 0))
            if rc != 0:
                raise RuntimeError(
                    f"Unexpected error copying directory: {resp.get('output', '')}"
                )
            return

        if not os.path.isfile(host_src):
            raise FileNotFoundError(
                f"Source path '{host_src}' is not a valid file or directory."
            )

        dest_stat = self._stat_path(dest_actual)
        dest_exists = dest_stat.get("exists")
        dest_is_dir = dest_stat.get("is_dir")
        dest_basename = posixpath.basename(dest_actual)
        if dest_is_dir or dest_display.endswith("/") or (
            not dest_exists and "." not in dest_basename
        ):
            target_path = posixpath.join(dest_actual, os.path.basename(host_src))
        else:
            target_path = dest_actual

        with open(host_src, "rb") as f:
            payload = base64.b64encode(f.read()).decode("ascii")
        script = """import base64, os
path = os.environ["MSWEA_PATH"]
payload = base64.b64decode(os.environ["MSWEA_CONTENT_B64"])
parent = os.path.dirname(path)
if parent:
    os.makedirs(parent, exist_ok=True)
with open(path, "wb") as f:
    f.write(payload)
"""
        resp = self._run_python(
            script,
            {
                "MSWEA_PATH": target_path,
                "MSWEA_CONTENT_B64": payload,
            },
        )
        rc = resp.get("returncode", resp.get("exit_code", 0))
        if rc != 0:
            raise RuntimeError(
                f"Unexpected error copying file: {resp.get('output', '')}"
            )

    def list_files(self, path: str | None = None) -> list[str]:
        target_display = path or "/workspace"
        try:
            actual_path = self._resolve_path(target_display)
        except ValueError:
            return []

        stat = self._stat_path(actual_path)
        if not stat.get("exists"):
            return []
        if stat.get("is_file"):
            return [self._display_from_actual(actual_path)]

        script = """import json, os
path = os.environ["MSWEA_PATH"]
items = [os.path.join(path, entry) for entry in os.listdir(path)]
print(json.dumps({"ok": True, "items": items}))
"""
        data = self._run_python_json(script, {"MSWEA_PATH": actual_path})
        if not data.get("ok"):
            return []
        return [self._display_from_actual(item) for item in data.get("items", [])]

    def copy_from(self, path: str):
        from pathlib import Path
        import tempfile

        try:
            actual_path = self._resolve_path(path)
        except ValueError as e:
            raise FileNotFoundError(f"Path not found: {path}") from e

        script = """import base64, io, json, os, zipfile
path = os.environ["MSWEA_PATH"]
if not os.path.exists(path):
    print(json.dumps({"ok": False, "error": "Path not found"}))
    raise SystemExit(0)
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for filename in files:
                file_path = os.path.join(root, filename)
                arcname = os.path.relpath(file_path, path)
                zipf.write(file_path, arcname)
    else:
        zipf.write(path, os.path.basename(path))
payload = base64.b64encode(buffer.getvalue()).decode("ascii")
print(json.dumps({"ok": True, "content_b64": payload}))
"""
        data = self._run_python_json(script, {"MSWEA_PATH": actual_path})
        if not data.get("ok"):
            raise FileNotFoundError(data.get("error", "Path not found"))

        payload = base64.b64decode(data.get("content_b64", ""))
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        temp_file.write(payload)
        temp_file.close()
        return Path(temp_file.name)
