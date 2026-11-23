# Install and use with Codex or Claude Code

Use the [starting prompts](../USER_PROMPTS.md) for a new workspace or an existing
project. The AI setup procedure distinguishes the software folder from the target
project and reviews existing work before adding capabilities.

For a new workspace, open the complete software folder; for an existing project,
open that project and provide the separate ESA software path. Ask the AI to follow
[SETUP.md](../SETUP.md). It needs no previous ESA context or global Genesis skill.
The following commands are the equivalent direct route. Use a new or deliberately
chosen project folder; do not extract an update over a working case.

## Windows, PowerShell

Run from the folder containing setup_esa.py and pyproject.toml:

~~~powershell
py -3.11 -I setup_esa.py plan --client codex
py -3.11 -I setup_esa.py install --client codex
~~~

## Linux, Bash

Run from the folder containing setup_esa.py and pyproject.toml:

~~~bash
python3.11 -I setup_esa.py plan --client codex
python3.11 -I setup_esa.py install --client codex
~~~

A compatible Python >=3.11 can be selected instead; 3.11 is the reference version.
Use --client claude for Claude Code, or both when both clients are wanted.
The helper creates .venv/ using the selected system Python and pip-installs this
local source plus full numerical/neural/card dependencies. It does not change
system Python packages. Python's venv and pip must be available (some Linux
installations supply venv in a separate OS package). No uv or shell activation
is required. If Python/venv is absent, the AI should explain the normal OS setup
and respect the user's authority for system changes.

Default CPU PyTorch provides a working fallback. For GPU use, prepare a private
environment following the actual hardware's PyTorch instructions, then choose
--env ABSOLUTE_ENV_PATH --reuse-env --torch preserve. Verify a real GPU operation;
setup's import checks do not qualify GPU training. --profile analysis is an
explicit smaller field-analysis runtime; full is the modeling/card default.
Numerical libraries can be much larger than this source folder; downloads and
long scientific work have no overall helper deadline. --probe-timeout only
bounds a small local MCP diagnostic.

For an existing case, keep the software folder separate and add
--project ABSOLUTE_PROJECT_PATH to the same commands. Existing README, unrelated
client choices, data, decisions and local knowledge are preserved. Existing
environments require deliberate reuse; another ESA version needs
[staged adoption](local_learning_and_updates.md), not an in-place upgrade.

For direct pip/CLI control, the helper performs the standard equivalents:
create a private venv with python -m venv, install compatible CPU torch when
needed, run that environment's python -m pip install "ABSOLUTE_SOURCE_PATH[full]",
then python -m element_surrogate.cli doctor --profile full and
python -m element_surrogate.cli init --project ABSOLUTE_PROJECT_PATH
--name my-magnet --client codex. It additionally checks source/installed bytes,
runs a vacuum control and verifies an actual MCP handshake/tool call. Using the
helper retains these extra checks and per-attempt logs.

## Client connection and first work

Open/trust this project in your installed, signed-in client. Accept its normal
project-tool permissions and reopen/reload if necessary. The project files are:

| Client | Tool connection | Workflow skills |
|---|---|---|
| Codex | .codex/config.toml | .agents/skills/ |
| Claude Code | .mcp.json | .claude/skills/ |

No global client settings or authentication files are changed. The AI should
check its MCP status and call the ESA inspect_file tool on a small setup document.
A successful helper test does not prove that the running AI loaded its tools.
Invoke $field-surrogate-workflow in Codex or /field-surrogate-workflow in Claude
Code after describing the scientific task; see the [manual](usage_manual.md).

Put original fields under inputs/fields/. Initialization also creates visible
data/, studies/, models/, reports/ and records/; README.md explains each.
Case ownership, reuse terms and access facts are determined by the user, not
inherited from the base-software license. No setup step uploads anything.

## Recovery and later changes

Read records/setup/latest.json and the linked attempt's result/logs. Repeat
install to resume an owned incomplete environment. Same-version cases are
verified without pip reinstallation or budget resets. Reuse the original
--env/--profile/--client choices for verify. If install.lock remains after a
reboot, inspect its host/PID and actual process before removing that one lock.

Source is visible in src/, but the active runtime is the installed non-editable
package. Reinstall and validate deliberate software changes. A moved/deleted
Python environment also requires a reviewed update of its absolute MCP command.
Normal project-scoped setup does not hide inherited global/parent client context;
a strict independence experiment needs an appropriate OS/account boundary.
Doctor/smoke/setup checks do not qualify new fields/models or prove portability
on platforms not actually tested. See [QUALIFICATION.md](../QUALIFICATION.md).
