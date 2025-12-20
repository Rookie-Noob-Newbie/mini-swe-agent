# Integrating OpenHands CodeActAgent (decision only)

Goal: keep **mini-SWE-agent's runner and docker exec path** unchanged, but swap the **decision layer** to OpenHands `CodeActAgent`. Commands still run via mini-SWE-agent's `Environment.execute` (docker exec), not via OpenHands runtime.

## High-level design

1. **Decision**: CodeActAgent plans the next shell command.
2. **Execution**: mini-SWE-agent `Environment` runs the command (e.g., `DockerEnvironment.execute`).
3. **Feedback**: stdout/stderr/returncode are fed back to CodeActAgent as the observation for the next step.

We do **not** start the OpenHands action-execution server or runtime container. All execution stays in the existing mini-SWE-agent docker exec flow.

## Minimal adapter you need to add

Create `src/minisweagent/integrations/codeact.py` with:

- A thin **RuntimeAdapter** that implements the minimal interface CodeActAgent calls:
  - `run_cmd(cmd: str) -> dict`: internally calls `env.execute(cmd)` and returns stdout/stderr/returncode.
  - `connect()/disconnect()`: no-op or lightweight.
  - If you disable editor/file tools (set `enable_editor=False`, `enable_llm_editor=False`, `enable_browsing=False` in the agent config), you only need command execution.
- A **CodeActWrapper** that:
  - Builds `LLMConfig` from mini-SWE-agent model config (can reuse OpenHands-style `config.toml` section).
  - Instantiates `CodeActAgent(llm=..., runtime=RuntimeAdapter(...), max_iterations=...)`.
  - Provides `run(instance)` that returns `(exit_status, result_string)` compatible with mini-SWE-agent's `save_traj/preds.json`.

Suggested shape (pseudo-code, adjust to actual imports):

```python
from dataclasses import dataclass

from minisweagent import Environment
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config.llm_config import LLMConfig


class RuntimeAdapter:
    def __init__(self, env: Environment):
        self.env = env

    def connect(self):  # optional
        return None

    def disconnect(self):  # optional
        return None

    def run_cmd(self, cmd: str) -> dict:
        out = self.env.execute(cmd)
        return {
            "stdout": out.get("stdout", ""),
            "stderr": out.get("stderr", ""),
            "returncode": out.get("returncode", 0),
        }


@dataclass
class CodeActResult:
    exit_status: str
    result: str


class CodeActWrapper:
    def __init__(self, env: Environment, llm_config: dict, max_steps: int = 100):
        self.runtime = RuntimeAdapter(env)
        self.llm_cfg = LLMConfig(**llm_config)
        self.max_steps = max_steps

    def run(self, task: str) -> CodeActResult:
        agent = CodeActAgent(
            llm=self.llm_cfg,
            runtime=self.runtime,
            max_iterations=self.max_steps,
            enable_browsing=False,
            enable_editor=False,
            enable_llm_editor=False,
        )
        status, result = agent.run(task)
        return CodeActResult(exit_status=status, result=result)
```

## Hook into the SWE-bench runner

In `src/minisweagent/run/extra/swebench.py` inside `process_instance`:

- If config says `environment_class: codeact`, then:
  - Still build the `env = get_sb_environment(...)` (docker exec).
  - Instantiate `CodeActWrapper(env, llm_config=config["model"], max_steps=config["agent"]["max_steps"])`.
  - Call `res = wrapper.run(task)`; set `exit_status, result = res.exit_status, res.result`.
- Else keep the existing `DefaultAgent` path unchanged.

Outputs (`traj.json`, `preds.json`) stay identical; only the decision-maker changes.

## Config example (new preset)

Use built-in preset `src/minisweagent/config/extra/swebench_codeact.yaml` (pass via `--config`):

```yaml
model:
  model_name: llm.glm46_eval_1   # use your OpenHands-compatible model section
environment:
  environment_class: codeact     # triggers the adapter branch
  image: ""                      # unused here; get_sb_environment will fill from instance
agent:
  max_steps: 100
run:
  env_startup_command: ""        # optional
```

Run:

```bash
mini-extra swebench --subset verified --split test \
  --config src/minisweagent/config/extra/swebench_codeact.yaml \
  --workers 1 --output tmp/mini-codeact-out
```

## Testing checklist

1. **Dry-run without docker**: mock `Environment.execute` to return a fixed stdout; confirm CodeActAgent loops and stops.
2. **Single SWE-bench instance with docker**: verify commands land via `env.execute`, no OpenHands runtime/container is started.
3. **Concurrency**: ensure each worker gets its own `Environment` instance; adapter is lightweight and stateless.

## Troubleshooting

- If CodeActAgent asks for editor/browsing tools, disable them via constructor flags (see wrapper above).
- If LLM config parsing fails, mirror the exact dict OpenHands expects (`model_name`, `api_key`, `base_url`, etc.).
- Keep return payload from `run_cmd` consistent (`stdout`, `stderr`, `returncode`) so the agent’s parsing is stable.
