# AI-assisted setup instructions

You are setting up the supplied Element Surrogate Agent (ESA), a specialist
package for single-magnet field-to-transfer-map research. The running Codex or
Claude Code client supplies the LLM, tool execution and permissions. ESA supplies
physics knowledge, numerical tools, workflows and bundled Genesis/benchmark
resources. This document works without a previous ESA conversation or project.

## Goal and boundaries

Reach a verified project-local installation with few user interventions. Read
START_HERE.md, README.md and QUALIFICATION.md from the ESA software folder.
Use this document as the setup procedure; the software AGENTS.md distinguishes
user science from base-software maintenance. Its uv commands are developer
checks, not an end-user prerequisite. Do not recreate the tracker, benchmark
pipeline, cards or agent framework. The helper reuses the released initializer.

Preserve original inputs, README, unrelated client configuration, accepted
decisions and local knowledge. Do not change global client settings or install
libraries into global Python. Installing this software assigns no ownership or
license to the user's future data/models. Do not request credentials in chat.
No upload, scientific data generation, training or model publication is setup.

## Choose the project and scope

The **ESA software folder** contains this SETUP.md, setup_esa.py, src/ and skills/.
The **target project** is where the user works and where setup writes its records
and client connection. They are the same folder for a new extracted workspace;
for an existing project they can be different. Resolve this document's relative
source/document paths from the software folder, not from the current shell folder.

For an existing project, first read its instructions, current status and relevant
scientific evidence. Identify which ESA capabilities serve the user's goal.
Reuse existing fields, simulation code, data, models and reports when appropriate;
record their actual provenance rather than describing them as ESA-generated.
Accepted conventions and local knowledge remain authoritative until evidence or
the user's request calls for review. Explain important conflicts before changing
established choices; continue work that does not depend on resolving them.
Adding ESA is not a request to repeat generation/training or reorganize the project.

Use the existing project's organization. For a multi-magnet repository, select the
relevant single-element study as the case location. The initializer adds standard
work/records directories; inspect name conflicts before mutation and preserve
existing artifacts. Refer to existing files rather than making unneeded copies.
The client must open the target whose configuration was written. If the target
is a study subfolder of the currently open repository, explain which folder to
open/reopen before confirming the live tool connection.
A guidance-only task needs no runtime installation. When tools are wanted, choose
the runtime profile for that work; installation registers all five skills and
individual skills can be used as needed.

Preserve the existing scientific environment. Use a separate private ESA
environment when compatibility is uncertain; reuse an existing environment only
after inspecting compatibility and confirming that its modification is within the
user's request. Do not upgrade a working scientific stack just to satisfy ESA.

## 1. Inspect the environment and show the plan

Identify which client you are using, the project directory, a working Python
>=3.11, available storage and intended CPU/GPU/remote resources. Explain the
download/runtime footprint qualitatively; do not invent an exact time or size.
Use Python 3.11 when available because it is the reference configuration.
If no compatible Python exists, explain the normal installation route and obtain
the user's authority for system-level installation. Do not add uv as a prerequisite.

The software-folder root is the default target project. It may have been renamed
and contain spaces. For another target, add --project ABSOLUTE_PROJECT_PATH to
every plan/install/verify command. Run setup_esa.py from the software folder or
use its full path. Select --env ABSOLUTE_ENV_PATH for a separate private runtime.
Use an explicit interpreter; there is no activation-script requirement.
Run the read-only plan, replacing the interpreter/client for this environment:

~~~text
python -I setup_esa.py plan --client codex
~~~

Use --client claude for Claude Code; both is available when requested.
The default profile is full: field tools, neural training/export and Genesis
cards. Analysis is an explicit lightweight choice, not a full modeling setup.
If an existing environment is appropriate, inspect it and use --env ABSOLUTE_PATH
--reuse-env deliberately. The default creates this project's .venv from the
selected Python installation. Existing differing ESA versions use the staged
update procedure in docs/local_learning_and_updates.md; do not overwrite or
reset an active case.

Local CPU is the working fallback. For an intended GPU setup, use the official
PyTorch instructions appropriate to the actual hardware to prepare the selected
environment, then use --torch preserve --reuse-env. Verify the GPU actually works;
a CUDA wheel or nvidia-smi alone is not a successful training check. If GPU
configuration is unavailable, explain the CPU/remote alternatives rather than
treating a GPU as a prerequisite.

Review the returned plan and resolve concrete conflicts. Reuse permissions
already given; do not ask the user to approve each ordinary project-local step.

## 2. Install and verify

Run the helper with the same project, environment and client choices:

~~~text
python -I setup_esa.py install --client codex
~~~

The helper snapshots the local runtime source and skills, creates/reuses the
selected private environment, updates pip there, and installs this local source
project and declared dependencies with pip. It initializes the existing
project-scoped skills/MCP connection, runs doctor and the numerical smoke control, and opens a real MCP session to discover/call a tool.
It never changes the global client configuration. CPU PyTorch is installed only
as a fallback; an already compatible installation is preserved.

The client may protect its own configuration directories. Use its normal
approval mechanism for the concrete project-only setup command. If the client
cannot write these files, give the user the same single helper command to run
from an owner terminal, then verify the result. Do not bypass a refusal through
another execution channel or change global permissions.

Commands may download large libraries. There is no overall installation or
scientific-job deadline. Use the client's normal background process/session
facilities and monitor records/setup/attempts/* logs. Do not kill an authorized
install merely because a single shell/UI call returns early. The 60-second
default probe bound applies only to a small local protocol diagnostic; use
--probe-timeout if a slow startup needs longer.

On failure, inspect the saved result/logs, explain the actual cause and repair
the environment/configuration within the user's authority. Do not weaken a
physics check or fabricate a successful result. Owned incomplete environments
can be resumed by rerunning install. Existing same-version cases are verified
without reinstalling or resetting them. To add a missing optional runtime
profile, install this local source with that declared extra into the exact
environment, then verify; do not upgrade an active ESA to another version.

If setup left an install.lock after an interrupted process, inspect its recorded
PID/host and the actual process before removing that lock. Do not remove a lock
belonging to a live setup or another machine.

## 3. Confirm the actual client connection

The helper's protocol test is necessary, but it is not proof that this running
LLM session has reloaded its tool catalog. Ask for ordinary project trust or a
client reopen only when needed. Check the client's MCP status. From THIS client,
use the ESA inspect_file tool on README.md (or another small setup document) and
confirm the response.

Invoke the installed field-surrogate-workflow skill when starting science:
$field-surrogate-workflow in Codex or /field-surrogate-workflow in Claude Code.
A client-specific selector is an interface detail; do not call scientific tools
before the actual modeling request and required source review.

Report setup as:
- runtime and direct MCP checks: passed/failed, citing the result path;
- this client's live connection: confirmed, or explicitly awaiting trust/reopen;
- available runtime profile and CPU/GPU evidence;
- next input expected from the user.

Do not say fully ready when the live client connection is still pending.
The numerical smoke control is not a Maxwell/field, tracking-domain, model
accuracy, production or independent-user qualification.

## Existing projects, source and later changes

For an existing project, follow the target/scope review above and keep the ESA
software separate. Never unzip over existing work. Existing files are protected
by the regular initializer. Conflicts require inspection and do not authorize
deleting local skills or knowledge. State/budget resets are not setup repairs.

The installed package is the active runtime. src/, skills/ and docs/ provide
visible source and reference material. Source changes require deliberate
reinstallation and validation before they affect the runtime. The helper compares
installed runtime/skill bytes with this source and records a snapshot per attempt.
This is consistency evidence, not authentication of the source publisher. Project lessons
belong in records under the local-learning procedure. Maintainer updates remain
separate, reviewed and reversible.

For manual pip/venv commands see docs/installation.md. To recheck a completed
setup with the same options:

~~~text
python -I setup_esa.py verify --client codex
~~~

On a later check, read records/setup/latest.json and its attempt plan to recover
the selected environment, client and profile. Reuse those options; do not assume
the active environment is .venv when the user chose another location.

Keep attempt logs and resolved dependency versions. Do not upload them without
the optional feedback procedure and exact-content consent.
