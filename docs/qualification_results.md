# Independent first-user onboarding results

**Result: both starting routes passed for ESA 0.2.0rc29.** These tests exercised
AI-assisted installation and adoption into existing work. They were completed
on 2026-11-21, America/New_York, and add interface evidence to the earlier
successful scientific campaigns. The [accumulated qualification](../QUALIFICATION.md)
also covers the five-million-particle real-field study, unchanged numerical
components, Windows checks and current cross-platform CI.

## Question and test conditions

Could a fresh Codex session follow the short user prompts, install the supplied
software and use its tools without the originating research repository, global
Genesis skills or previous case answers?

The test used native Linux Codex CLI `0.166.0-alpha.5.1`, Python 3.13.5 and separate
fresh project/client profiles. An OS filesystem boundary hid the normal home,
global skills, parent research projects and other test cases. Only the relevant
case and ESA source were supplied. Normal client protections remained enabled.
Existing sign-in was mounted read-only for client authentication; the observer
did not read or copy credentials. Built-in client system guidance remained.

The requested model was GPT-6.1 Sol with extra-high effort (`xhigh`). The available
catalog advertised that selection. Launch records attest the request; no separate
provider-side model-routing attestation was returned.

The VM and filesystem boundary were test apparatus. **Users do not need a VM,
Docker or this isolation setup to install ESA.**

ESA tools connect through the Model Context Protocol (MCP).

## Prompts and outcomes

The initial prompts were the two setup blocks in [USER_PROMPTS.md](../USER_PROMPTS.md),
unchanged except replacing the existing-project software path. The user's
instruction not to create a ZIP was appended. The observer supplied no hidden
solution, exact tool sequence or scientific-model choices.

| Route | Starting state | Observed outcome |
|---|---|---|
| New project | Clean software folder; no installed ESA or initialized case | Chose a private full CPU environment; installed the package; registered five skills; verified runtime/dependencies, numerical smoke and a direct MCP call; explained field input and next steps |
| Existing project | Synthetic 1 m drift implementation, two checked CSV rows, accepted mm/mrad conventions, a working frozen environment and an original project skill/client setting | Reviewed accepted work; chose a separate `cards` environment; added documentation/benchmark support; retained original scientific files, skill, conventions and frozen dependencies; original regression passed |

Each initial session correctly reported that the current client's live connection
still needed confirmation. The simulated user reopened the project and sent:

~~~text
I reopened this project. Please confirm ESA is available in this session and finish the setup check.
~~~

In both reopened sessions, native `element_surrogate.inspect_file` calls on the
project README succeeded and **99 ESA tools** were available. This is separate
from the installer's direct protocol probe. All four native turns exited
successfully. No scientific data generation or retraining was performed.

An observer hash comparison found no missing original project member. Only the
intended project MCP connection changed in its client configuration; its unrelated
setting remained. The original dependency freeze and regression were unchanged.
The existing-project session also checked local Genesis model/data guidance,
dataset introspection and the four pinned benchmark-builder assets. It did not
claim to have completed a scientific model/data card during setup.

## Observed time and runtime footprint

| Route | Initial native turn | Reopen turn | Private environment, regular-file bytes |
|---|---:|---:|---:|
| New project, full CPU | 231.4 s | 121.9 s | 1,490,853,840 |
| Existing project, cards | 303.2 s | 146.8 s | 465,846,115 |

These are whole-turn measurements, including LLM reading, installation, checks and
handoff. They are not pure installation times or promises for another machine.
Environment sizes exclude symlinks; downloads/caches require additional storage.
The software folder itself is much smaller. No matched workflow-efficiency or
surrogate inference-speed comparison is inferred from this setup exercise.

## Recoveries and interpretation

- The existing-project initializer first encountered a read-only client config.
  The agent retained that failed attempt and completed setup using the normal
  approval mechanism for the project-only command. Some clients may instead ask
  the user to run that command from an owner terminal.
- The extracted folders were not Git repositories and `rg` was absent. The
  sessions recovered with available tools. Neither Git nor `rg` is a prerequisite
  for end-user setup.
- The test cache mapping produced a pip ownership warning. Setup and dependency
  checks passed without changing global permissions.
- An initial observer transport failed before native testing. The corrected
  transport retained prompts and raw events outside the tested clients. Observer
  repairs did not supply answers to the agents.

No ESA distribution, case-backup or checkpoint archive was created. The installed
setuptools dependency included its own small upstream ZIP test fixture, verified
against its package RECORD; it was not an ESA-generated archive or shipping asset.

The original 315-file clean software copy was unchanged by testing. This GitHub
preparation subsequently updates public documentation and distribution metadata;
the tested runtime, setup helper and scientific skill bytes are preserved.
Raw prompts/events, failures, source identities and full local result records are
retained by the maintainer. They are summarized here without local machine paths,
account details, credentials or research data.

**Scope:** this report adds one manufactured case per entry route and one reopen
each. Linux supplied the test's strong filesystem isolation; Windows installation,
runtime and tool behavior were checked separately, including in GitHub CI. A
fresh Windows/Claude conversation would add client-interaction coverage. Earlier
scientific and software results continue to support unchanged components; they
are summarized in [qualification](../QUALIFICATION.md), alongside the actual
[physical/runtime scope](deployment.md).
