# Agent entry point

This repository holds the tool library, MCP server, skills and knowledge base of
an agent that builds neural-network surrogates of one magnet's particle transfer
map from its raw 3D field map. Read `README.md` for the current status and
`CONTRIBUTING.md` for the project rules; the rules below are the ones an agent
working here must not break.

End-user installation follows the source-root SETUP.md. The uv commands
below are developer checks, not an end-user installation prerequisite.

## User case versus base software

This folder can be both the supplied software and an initialized user workspace.
For a user's modeling request, use skills/field-surrogate-workflow/SKILL.md (or its
installed client copy). Inputs belong in inputs/, generated labels in data/,
studies in studies/, models in models/, reports in reports/, and decisions/local
knowledge in records/. These are legitimate user artifacts; do not forbid or
delete them because the generic base software ships without facility data.
Ownership, sharing and terms of user artifacts are the user's decisions.

The following development rules govern changes to base software, shared skills,
knowledge and distribution. They do not prohibit user case artifacts in those
work folders. Read SHIPPING.md before distributing a software-only copy; never
bundle a used scientific workspace, local environments, client accounts or Git
history into that copy. Preserve project lessons when adopting managed updates.

## Base-software development rules

- **No facility data in the shipped base software.** Never add field maps, trained models, checkpoints,
  particle data, magnet case records, process-variable names, or numbers taken
  from one facility's magnets, even as test fixtures. Tests build synthetic
  fields at run time. Known-answer checks against a facility's prior work run
  from that facility's own repository against the installed package.
- **Physics in tested tools; judgement in skills.** A rule that must always hold
  is code with a test. Skills hold the procedure and say when to ask the user.
- **No magnet-specific defaults.** A number from one magnet becomes an argument
  or a knowledge-base entry with its evidence.
- **Summaries, not arrays.** Tools return JSON summaries with units in the key
  names and the paths of files they wrote; raw arrays never reach the language
  model.
- **Define each tool once.** Register it with `@tool` in `element_surrogate/tools.py`;
  the MCP server generates the tool, its schema and description, and records every
  call. Do not write tools in the server.
- **One region.** Checks and fits take `element_surrogate.region.Region`; do not add
  another way of saying where.
- **Findings state their criterion.** A check that flags something returns findings,
  each naming its criterion: a statistic of the map itself or a tolerance the user
  stated. A new check, or a new kind of defect, gets a case in
  `element_surrogate/corpus.py`, so the runner shows it flagged and shows its false
  alarms on clean fields.
- **Measured, not assumed.** A tool that reports a number from a fit or a
  discretization also reports how much it moves under refinement (coarsening,
  higher order); a reading of a statistic states the signatures it was checked on.
- **Knowledge ships general.** Shipped knowledge entries (`src/element_surrogate/kb/`)
  cite only public sources, derivations or this package's tests, mark unverified
  sources, and new shared entries start as `proposed` pending maintainer review.
  Private case entries may be activated by explicit user decision or validated
  agent judgment under a recorded, revocable local delegation. Neither path
  changes shared knowledge. A method entry names its tested implementation.
- **One magnet at a time.** Lattices, overlapping fields of neighbouring magnets,
  and composition of maps are out of scope.

## Before a public release

- Keep the user-supplied base-software owner/maintainer contact consistent in
  `pyproject.toml`, `cards/agent_card.md` and `SECURITY.md`. The designated
  contact is Kilean Hwang (FRIB), hwang@frib.msu.edu. Do not infer
  institutional copyright or sponsorship from this maintenance attribution.
- Future users determine ownership and terms of their project software,
  data and models; never inherit base-agent ownership or license into them.
- Facility-data governance does not block distributing this software when
  no such data ship. Apply only established obligations; invent no approval.
- Add an acknowledgment of Genesis Mission support to `README.md` only if the
  work is funded under the Genesis Mission.
- Review every card's `sensitivity`.

## Verification

Benchmark studies: read `skills/surrogate-benchmark/SKILL.md` and its protocol
guide before designing or running one. Keep caller-attested user decisions
distinct from authenticated identity, and trusted adapter execution distinct
from a security sandbox. No tool creates user approvals.

```bash
uv sync --group dev --group test
uv run pytest
uv run ruff check . && uv run ruff format . --check
uv lock --check
make ext && make conformance
```


For the local field-to-model workflow, read `skills/field-surrogate-workflow/SKILL.md`
and its file-contract reference. Synthetic physical controls do not qualify a real
magnet or authenticate client approvals. Never ship generated case artifacts.
