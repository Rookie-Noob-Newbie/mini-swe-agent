#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import copy
import json
import os
import random
import re
import shutil
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import logging
import typer
import yaml
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger, set_console_log_level
from minisweagent.integrations.codeact import CodeActRunner

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


_OUTPUT_FILE_LOCK = threading.Lock()


_SWE_BENCH_DEFAULT_PROMPT_TEMPLATE = """<uploaded_files>
/workspace/{{ workspace_dir_name }}
</uploaded_files>

I've uploaded a python code repository in the directory {{ workspace_dir_name }}. Consider the following issue description:

<issue_description>
{{ instance.problem_statement }}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the /workspace/{{ workspace_dir_name }} directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices or conventions in them.
   1.2 Hightlight message errors, method names, variables, file names, stack traces, and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Hightlight any best practices to take into account when testing and fixing the issue

Phase 2. RUNNING: install and run the tests on the repository
   2.1 Follow the readme
   2.2 Install the environment and anything needed
   2.2 Iterate and figure out how to run the tests

Phase 3. EXPLORATION: find the files that are related to the problem and possible solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, select the most likely location to fix the issue.

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.
   4.1 Look at existing test files in the repository to understand the test format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 Run the reproduction script to confirm you are reproducing the issue.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests related to the modified code to ensure you haven't broken anything.

8. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit {{ instance.base_commit }}.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity.
"""

_DEFAULT_MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "5"))
_DEFAULT_TIMEOUT_SECONDS = int(os.getenv("EVAL_TIMEOUT_SECONDS", str(8 * 60 * 60)))


class _EvalAbort(Exception):
    pass


def _should_skip_maximum_retries() -> bool:
    return os.getenv("EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED", "false").lower() == "true"


def _log_maximum_retries_exceeded(output_dir: Path, instance_id: str, error: str) -> None:
    retries_path = output_dir / "maximum_retries_exceeded.jsonl"
    entry = {
        "instance_id": instance_id,
        "error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with _OUTPUT_FILE_LOCK:
        with retries_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def _log_maximum_retries_notice(output_dir: Path) -> None:
    retries_path = output_dir / "maximum_retries_exceeded.jsonl"
    if retries_path.exists():
        logger.info("ATTENTION: Some instances reached maximum error retries and were skipped.")
        logger.info(f"These instances are listed in: {retries_path}")
        logger.info(
            "Fix these instances and run evaluation again with EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=false"
        )


def _is_fatal_runtime_error(error: str | None) -> bool:
    if not error:
        return False
    fatal_errors = [
        "AgentRuntimeTimeoutError",
        "AgentRuntimeUnavailableError",
        "AgentRuntimeDisconnectedError",
        "AgentRuntimeNotFoundError",
    ]
    return any(err in error for err in fatal_errors)


class _ThreadAllowlistFilter(logging.Filter):
    def __init__(self, allowed_threads: set[int]) -> None:
        super().__init__()
        self._allowed_threads = allowed_threads

    def allow_thread(self, thread_id: int | None) -> None:
        if thread_id is None:
            return
        self._allowed_threads.add(thread_id)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread in self._allowed_threads


@dataclass
class _InstanceRunResult:
    exit_status: str
    result: str
    extra_info: dict | None
    agent: DefaultAgent | None
    model_name_for_output: str


def _get_swebench_workspace_dir_name(instance: dict) -> str:
    repo = instance.get("repo")
    version = instance.get("version")
    if repo and version:
        return f"{repo}__{version}".replace("/", "__")
    return str(instance.get("instance_id", "workspace"))


def build_openhands_swebench_instruction(instance: dict) -> str:
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    instruction = Template(_SWE_BENCH_DEFAULT_PROMPT_TEMPLATE).render(
        instance=instance,
        workspace_dir_name=workspace_dir_name,
    )
    if os.getenv("RUN_WITH_BROWSING", "false").lower() == "true":
        instruction += '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
    return instruction


def setup_openhands_workspace(env: Environment, instance: dict) -> str:
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    cmd = (
        "mkdir -p /workspace && "
        "rm -rf /workspace/* && "
        f"cp -r /testbed /workspace/{workspace_dir_name}"
    )
    out = env.execute(cmd)
    if out.get("returncode", 0) != 0:
        raise RuntimeError(f"Failed to prepare /workspace: {out}")
    if hasattr(env, "config") and hasattr(env.config, "cwd"):
        env.config.cwd = f"/workspace/{workspace_dir_name}"
    return workspace_dir_name


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_class = env_config.get("environment_class", "docker")
    if env_class == "codeact":
        # CodeAct uses CodeActAgent for decisions but still runs commands via docker exec.
        env_class = "docker"
        env_config["environment_class"] = env_class
    else:
        env_config["environment_class"] = env_class
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = "docker://" + image_name
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def _run_instance_once(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    instance_dir: Path,
    *,
    env_holder: dict[str, Environment] | None = None,
) -> _InstanceRunResult:
    use_codeact = config.get("environment", {}).get("environment_class") == "codeact"
    model = None if use_codeact else get_model(config=config.get("model", {}))
    task = build_openhands_swebench_instruction(instance)

    progress_manager.update_instance_status(instance["instance_id"], "Pulling/starting docker")
    env = get_sb_environment(config, instance)
    if env_holder is not None:
        env_holder["env"] = env
    workspace_dir_name = setup_openhands_workspace(env, instance)

    agent: DefaultAgent | None = None
    extra_info: dict | None = None

    if use_codeact:
        progress_manager.update_instance_status(instance["instance_id"], "CodeAct: starting")
        logger.info(f"[CodeAct] Starting instance {instance['instance_id']}")
        runner = CodeActRunner(
            env=env,
            llm_config=config.get("model", {}),
            max_steps=config.get("agent", {}).get("max_steps", 100),
            run_id=instance["instance_id"],
            repo_path=f"/workspace/{workspace_dir_name}",
            base_commit=instance.get("base_commit"),
        )
        res = runner.run_instance(
            task,
            progress_callback=lambda msg, iid=instance["instance_id"]: progress_manager.update_instance_status(iid, msg),
        )
        exit_status, result = res.exit_status, res.result
        progress_manager.update_instance_status(instance["instance_id"], f"CodeAct: {exit_status}")
        if res.steps_path:
            try:
                shutil.copy(res.steps_path, instance_dir / "codeact_steps.jsonl")
            except Exception as e:
                logger.error(f"Failed to copy steps log: {e}", exc_info=True)
        if res.history_path:
            try:
                shutil.copy(res.history_path, instance_dir / "codeact_history.jsonl")
            except Exception as e:
                logger.error(f"Failed to copy history log: {e}", exc_info=True)
    else:
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance["instance_id"],
            **config.get("agent", {}),
        )
        exit_status, result = agent.run(task)

    model_name_for_output = (
        model.config.model_name if model is not None else config.get("model", {}).get("model", "unknown")
    )
    return _InstanceRunResult(
        exit_status=exit_status,
        result=result,
        extra_info=extra_info,
        agent=agent,
        model_name_for_output=model_name_for_output,
    )


def _run_instance_with_timeout(
    *,
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    instance_dir: Path,
    timeout_seconds: int | None,
    thread_filter: _ThreadAllowlistFilter,
    env_holder: dict[str, Environment] | None,
) -> tuple[_InstanceRunResult | None, Exception | None, str | None, bool]:
    if timeout_seconds is None:
        try:
            res = _run_instance_once(
                instance,
                output_dir,
                config,
                progress_manager,
                instance_dir,
                env_holder=env_holder,
            )
            return res, None, None, False
        except Exception as e:
            return None, e, traceback.format_exc(), False

    result_holder: dict[str, _InstanceRunResult] = {}
    error_holder: dict[str, Exception] = {}
    traceback_holder: dict[str, str] = {}
    done = threading.Event()

    def _target() -> None:
        try:
            result_holder["result"] = _run_instance_once(
                instance,
                output_dir,
                config,
                progress_manager,
                instance_dir,
                env_holder=env_holder,
            )
        except Exception as e:
            error_holder["error"] = e
            traceback_holder["traceback"] = traceback.format_exc()
        finally:
            done.set()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread_filter.allow_thread(thread.ident)
    if not done.wait(timeout_seconds):
        return None, None, None, True
    if error_holder:
        return None, error_holder.get("error"), traceback_holder.get("traceback"), False
    return result_holder.get("result"), None, None, False


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWEBench instance."""
    # work on a private copy so per-instance mutations (e.g., environment_class rewrite) don't leak across threads
    config_base = copy.deepcopy(config)
    instance_base = copy.deepcopy(instance)
    instance_id = instance_base["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    progress_manager.on_instance_start(instance_id)

    # per-instance file log: only records emitted from this worker thread
    per_instance_handlers: list[logging.Handler] = []
    allowed_threads = {threading.get_ident()}
    thread_filter = _ThreadAllowlistFilter(allowed_threads)
    handler = logging.FileHandler(instance_dir / "minisweagent.log")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    handler.addFilter(thread_filter)
    per_instance_handlers.append(handler)
    for logger_name in ("minisweagent", "openhands"):
        logging.getLogger(logger_name).addHandler(handler)
    logging.getLogger().addHandler(handler)

    agent: DefaultAgent | None = None
    extra_info: dict | None = None
    exit_status = "error"
    result = ""
    model_name_for_output = config_base.get("model", {}).get("model", "unknown")
    max_retries = _DEFAULT_MAX_RETRIES
    timeout_seconds = _DEFAULT_TIMEOUT_SECONDS
    runtime_failure_count = 0
    write_outputs = True

    try:
        for attempt in range(max_retries + 1):
            progress_manager.update_instance_status(
                instance_id, f"Attempt {attempt + 1}/{max_retries + 1}"
            )
            config_attempt = copy.deepcopy(config_base)
            instance_attempt = copy.deepcopy(instance_base)
            env_holder: dict[str, Environment] = {}

            res, error, tb, timed_out = _run_instance_with_timeout(
                instance=instance_attempt,
                output_dir=output_dir,
                config=config_attempt,
                progress_manager=progress_manager,
                instance_dir=instance_dir,
                timeout_seconds=timeout_seconds,
                thread_filter=thread_filter,
                env_holder=env_holder,
            )

            if timed_out:
                exit_status = "timeout"
                result = f"Timeout after {timeout_seconds} seconds"
                extra_info = {"traceback": f"Timeout after {timeout_seconds} seconds"}
                try:
                    cleanup = getattr(env_holder.get("env"), "cleanup", None)
                    if callable(cleanup):
                        cleanup()
                except Exception:
                    pass
                break

            if error is None and res is not None:
                exit_status = res.exit_status
                result = res.result
                extra_info = res.extra_info
                agent = res.agent
                model_name_for_output = res.model_name_for_output
                break

            error = error or RuntimeError("Unknown error")
            tb = tb or traceback.format_exc()
            error_msg = f"{type(error).__name__}: {error}"
            if attempt == max_retries:
                if _should_skip_maximum_retries():
                    _log_maximum_retries_exceeded(output_dir, instance_id, error_msg)
                    exit_status = "error"
                    result = f"Maximum retries ({max_retries}) reached: {error}"
                    extra_info = {"traceback": tb}
                    break
                raise _EvalAbort(
                    f"Maximum error retries reached for instance {instance_id}"
                ) from error

            if _is_fatal_runtime_error(error_msg):
                runtime_failure_count += 1
                logger.error(
                    f"Runtime disconnected error detected for instance {instance_id}, runtime failure count: {runtime_failure_count}"
                )

            logger.error(
                f"Error in instance [{instance_id}]: {error_msg}. Retrying... (attempt {attempt + 1} of {max_retries})",
                exc_info=error,
            )
            time.sleep(5)
    except _EvalAbort:
        write_outputs = False
        raise
    finally:
        if write_outputs:
            save_traj(
                agent,
                instance_dir / f"{instance_id}.traj.json",
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                instance_id=instance_id,
                print_fct=logger.info,
            )
            update_preds_file(output_dir / "preds.json", instance_id, model_name_for_output, result)
            progress_manager.on_instance_end(instance_id, exit_status)
        # remove per-instance handlers so reused threads don't leak logs across instances
        for h in per_instance_handlers:
            for logger_name in ("minisweagent", "openhands"):
                try:
                    logging.getLogger(logger_name).removeHandler(h)
                except Exception:
                    pass
            try:
                logging.getLogger().removeHandler(h)
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option( builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(
        output_path / "minisweagent.log",
        extra_loggers=("openhands",),
    )
    set_console_log_level(
        logging.WARNING,
        "root",
        "minisweagent",
        "openhands",
        "httpcore",
        "httpx",
        "asyncio",
    )

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    config_path = get_config_path(config_spec)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except _EvalAbort as e:
                logger.error(str(e))
                for pending in futures:
                    if not pending.running() and not pending.done():
                        pending.cancel()
                raise
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager): instance[
                    "instance_id"
                ]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)
    _log_maximum_retries_notice(output_path)


if __name__ == "__main__":
    app()
