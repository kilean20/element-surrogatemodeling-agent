# Contributing to element-surrogate-agent

Thank you for contributing. This guide is adapted from AI-ModCon's
[BaseTemplate](https://github.com/AI-ModCon/BaseTemplate).

## Code of Conduct

This project and everyone participating in it is governed by our
[Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

## Project rules

These hold for changes to the shared base software. User case fields, particles,
models and records belong in the initialized work folders and are not software
contributions or shipping assets. See AGENTS.md and SHIPPING.md.

For base-software contributions:

- **No facility data.** Do not add field maps, trained models, checkpoints,
  particle data, magnet case records, process-variable names, or numbers taken
  from one facility's magnets. Tests build synthetic fields at run time.
- **Physics lives in tested tools.** Anything that must always hold (units,
  sign conventions, Maxwell tests, loss filtering) is code with tests, not an
  instruction in a skill or an entry in the knowledge base.
- **No magnet-specific defaults.** A number that came from one magnet (a length,
  an aperture, a threshold) becomes an argument, or a knowledge-base entry with
  its evidence.
- **Tools return summaries.** Every tool returns a JSON summary whose numeric
  keys carry their units, plus the paths of files it wrote. Tools never return
  raw field or particle arrays.
- **Each tool is defined once.** Register it with `@tool` in
  `element_surrogate/tools.py`; the MCP server generates the tool from it and
  records every call so that it can be replayed.
- **One region.** Checks and fits take `element_surrogate.region.Region`.
- **Findings state their criterion,** a statistic of the map or a tolerance the user
  stated; a new check or defect kind gets a case in `element_surrogate/corpus.py`.
- **Knowledge entries.** Follow `src/element_surrogate/kb/README.md`: public sources,
  derivations or package tests only, unverified sources marked, and status
  `proposed` pending maintainer review for shared releases. Private case entries
  can use explicit user approval or validated local delegation; neither path
  promotes an entry into the shared KB. See the local-learning/update guide.

## How to Contribute

### Reporting Bugs

Check the existing issues first. When you open a bug report, include the steps
that reproduce the problem, what you observed, what you expected, and your
environment (operating system, Python version, package version). Include the AI
client/version and selected ESA runtime profile for setup issues. Use a minimal
sanitized example; keep credentials, private field/particle data and unreviewed
session logs out of public issues. Security concerns follow SECURITY.md.

### Suggesting Enhancements

Open an issue describing the problem, the proposed behaviour, and why it would
be useful.

### Pull Requests

- Follow the style guidelines below.
- Add tests for any new functionality.
- Avoid platform-dependent code; CI runs on Linux, Windows and macOS.
- End all files with a newline.

## Development Setup

```bash
uv sync --group dev --group test
uv run pytest
uv run ruff check . && uv run ruff format .
make ext && make conformance   # AI-ModCon card check and skill validation
```

## Style Guidelines

- PEP 8, enforced with Ruff (`uv run ruff check .`, `uv run ruff format .`).
- Type hints on public functions.
- Google-style docstrings on public modules, classes and functions.
- Commit messages in the imperative mood, first line at most 72 characters.

## Guidelines for AI/LLM-Assisted Contributions

From AI-ModCon's BaseTemplate:

- **Remain accountable for all your outputs and decisions.**
   Individuals remain fully responsible and accountable for the accuracy, quality, appropriateness, and consequences of their work. Use of AI does not transfer this responsibility to the AI model, agent, or other tool.
- **Understand your work.**
   Regardless of how code or PR was produced, this project requires that authors illustrate a thorough understanding of any proposed changes. You must review such code line-by-line; it is your responsibility to ensure that it is correct, and that it does not breach copyright. Always critically engage with AI outputs, do not trust them implicitly. AI-assisted code, analysis, and artifacts must be tested and validated at a level appropriate to their impact. Authors are responsible for ensuring that generated code is correct, secure, maintainable, non-obfuscated, appropriately scoped, documented, and reproducible where relevant.
- **Disclose AI-generated or AI-assisted work.**
   If AI/LLM tools were primarily used to generate code or artifacts, this should be clearly indicated in the PR.
- **Use of AI to review PRs.**
   All PRs must be reviewed by a human reviewer. An LLM review may be used in addition to a human reviewer since this can help spot issues that a human may have missed, but this should not be the sole reviewer. The human reviewer should be fully accountable and responsible for the review feedback or comments (see 1).
- **Proprietary or personal information.**
   For this project, proprietary or personal information should never be sent to code generators or AI tools.
- **Be transparent, assume goodwill, and share what you learn.**
   Contributors should be open about relevant AI use, disclose details of AI use as appropriate to the project, engage constructively with colleagues, and share experiences and lessons learned with the project.

## Questions?

Open an issue with the label `question`.
