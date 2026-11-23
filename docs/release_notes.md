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

This is a supervised research preview. Setup success is not real-field or model
qualification. No fresh native Claude/Windows LLM, GPU/HPC, general unattended
production or live Google authorization/upload qualification is claimed. Each
scientific case needs its own field/tracking convergence, coordinate/reference,
domain, model-quality and timing evidence.

## Feedback and ownership

Report reproducible setup issues or suggestions through GitHub Issues with a
minimal sanitized example. Do not attach credentials, private field/particle
data or unreviewed session logs. Security concerns use [SECURITY.md](../SECURITY.md).
Optional shared scientific feedback requires exact-content review/authority;
local learning works without a cloud connection.

Base-software maintainer: Kilean Hwang, FRIB, hwang@frib.msu.edu.
Software license: Apache-2.0 with third-party notices retained. Users choose
ownership and terms for their own artifacts.
