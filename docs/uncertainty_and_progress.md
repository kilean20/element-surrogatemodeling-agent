# Keep useful research moving when facts are unavailable

Version 0.2.0rc24 adds structured, nonblocking `warnings` to the existing
readiness/review responses: issue, evidence and useful next action. These
advisories do not change eligibility or replace real execution errors.
Current-to-field calibration is [external by default](physics_controls.md).

Version0.2.0rc23 adds a progress-preserving procedure and optional
`write_scientific_uncertainty_review` tool. It adds no prerequisite to existing
generation, study or export requests. No numerical kernel/check is relaxed.

ESA should inspect evidence, ask useful questions once and reuse explicit
answers/delegation. When the user cannot provide an aperture, improved source
field or another fact, it should compare feasible alternatives and tell the user
the supported decision and limitations. A consequential uncertainty affects
meaning/scope/claims; it is not automatically a reason to halt.

Examples include a numerical research region instead of an unknown physical
aperture, fixed-shape field/rigidity controls without an amperes calibration, or
continued transverse/path work while actual arrival time is undefined. Do not
silently change coordinates, explicit accuracy goals or the user's intent.
Unsupported requested outputs remain unsupported. Use the existing authority
and numerical checks; a review note grants no permissions or physical validity.

For substantial decisions, obtain the installed example:

```bash
element-surrogate request-example uncertainty-review --output uncertainty-request.json
```

Replace example prose with actual analysis, evidence/source references,
alternatives, plan, limitations and existing authority. Place the request and
report in visible `records/uncertainties/` and link from the report/resume.
The tool returns the report hash and question-pacing guidance; raw field/particle
arrays are not part of it. It does not contact the user or judge the physics.
The example is not a valid scientific qualification or user approval.

The full decision procedure is bundled in the workflow skill's
`references/uncertainty_and_progress.md`; no global skill or developer memory is
required. Read saved decisions when resuming instead of asking answered
questions again. There is no fixed question count, answer deadline or requirement
to know species for a signed-rigidity-only4D map. A local model can be useful with
unknown physical/governance facts; do not certify those unknown claims.

Numerical qualification, source/stencil identity, invalid data, resource ceilings
and explicit authorization still apply. A true inability to define an output
does not justify fabricating its labels: explain alternatives and finish other
authorized useful work. Optional feedback delivery never blocks modeling.

rc22 evidence remains pinned to its source/wheel. A new source/skill release has
a different implementation identity; do not re-stamp old qualifications as new
ones. Restore old numerical work with its pinned runtime or deliberately requalify
an updated case. Existing request schemas and approvals were not changed.

Manufactured library/MCP/generation tests verify these software boundaries.
Observed LLM follow-through requires its own fresh conversation evidence; no
universal autonomous behavior or new physical-field acceptance is asserted.
