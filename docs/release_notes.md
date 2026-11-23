# ESA 0.2.0rc29 — supervised research preview

This is a release-note draft for the maintainer. It does not claim that a GitHub
release has already been published.

ESA supplies local numerical tools, five workflow skills, physics guidance,
comparable model-study procedures and bundled Genesis resources for single-magnet
surrogate modeling with Codex or Claude Code.

## Start

Download this version's **Source code (zip)** from GitHub and extract it. Open
the folder in your signed-in AI client, then use [USER_PROMPTS.md](../USER_PROMPTS.md).
The same guide covers adding ESA to an existing project while keeping accepted
work. Python 3.11 or newer with venv/pip is required. Setup installs private
dependencies; uv, Docker and global Genesis skills are not prerequisites.

## Evidence

- Earlier independent real-field qualification generated five million retained
  particles, completed 54 fits and met its held-out accuracy and speed goals.
  Hash comparison confirms the current numerical/training/benchmark engine is
  unchanged from that campaign.
- Published-code GitHub CI passed the Windows/Linux/macOS test matrix and
  full-runtime/source-folder setup jobs on Windows and Linux.
- 75 focused installed Windows tests passed, along with a fresh full CPU
  source-folder installation and real local MCP checks.
- Independent native Linux/Codex onboarding passed for both a new project and
  an existing manufactured project, including actual tool calls after reopening.
- Existing scientific files, conventions, original skill and frozen dependencies
  were preserved; the intended project tool connection was added.
- Bundled runtime/skills and pinned Genesis/benchmark resources were checked.

The current publication preparation changes documentation and distribution
metadata while preserving the tested numerical runtime, setup helper and
scientific guidance. Read [QUALIFICATION.md](../QUALIFICATION.md) and the
[onboarding report](qualification_results.md) for exact scopes and recoveries.

## Limits

This is a supervised research preview. Earlier scientific campaigns remain
evidence for unchanged components; the latest setup tests complement them.
Platform/conversation coverage is recorded separately from practical limits.
Each new field/model needs suitable convergence, coordinate/reference, domain,
accuracy and timing checks. Implemented-physics/resource limits and optional
Google feedback status are described in QUALIFICATION.md.

## Feedback and ownership

Report reproducible setup issues or suggestions through GitHub Issues with a
minimal sanitized example. Do not attach credentials, private field/particle
data or unreviewed session logs. Security concerns use [SECURITY.md](../SECURITY.md).
Optional shared scientific feedback requires exact-content review/authority;
local learning works without a cloud connection.

Base-software maintainer: Kilean Hwang, FRIB, hwang@frib.msu.edu.
Software license: Apache-2.0 with third-party notices retained. Users choose
ownership and terms for their own artifacts.
