# Bound field-readiness assessment

Read before preparing a new generation plan. This is a completeness/identity
contract for trusted local research, not a physical pass threshold or identity
authentication. User decisions remain separate.

1. Save the transport specification with the exact source, reference and support.
   `collect_field_readiness(path, output_path, shells, end_tolerance)` computes
   raw-grid node, Maxwell, harmonic-node and field-end diagnostics. Choose shell
   count explicitly. The optional end tolerance is a declared study criterion.
   Optional `working_memory_bytes` (default 256 MiB) targets numerical working
   arrays. Exact global statistics and harmonic scalar maps use owned disk
   scratch; `scratch_directory` defaults to the evidence output's parent.
   Check free disk and use a disk-backed directory. Process RSS and virtual
   file mappings are separate from the working-array budget. Lowering this
   budget changes processing blocks, not source resolution or diagnostic scope.
   On Linux, RLIMIT_AS includes virtual source and scratch file mappings. It is
   not a resident-memory limit; do not set it equal to this array budget or
   infer required physical RAM from an address-space failure. Budget those
   mappings separately and measure process memory and available system memory.
2. Inspect the evidence and its issue inventory. Prepare a review JSON with
   `schema: element_surrogate.field_readiness_review.v1`, its canonical
   `evidence_sha256`, `categories`, and `issue_resolutions`.
3. `assess_field_readiness` validates the source/support and emits an assessment.
   Embed that complete assessment as `field_readiness` in the generation plan
   and set `field_readiness_sha256` to its canonical identity. Changing the plan
   requires fresh protocol-bound user attestations as before.

The six categories are `source_and_description`, `units_frame_polarity`,
`grid_integrity`, `vacuum_applicability_and_consistency`, `field_coverage`, and
`material_and_domain`. Each has `status` (`reviewed`, `not_applicable`,
`unresolved`, `rejected`), a nonempty `rationale`, and `evidence_refs` identifying
`source_summary` or named diagnostics in the evidence. Scientific interpretation
must explain the acceptance criterion, applicability and limitations; a small
residual does not prove source identity, calibration or material clearance.

Every issue ID emitted by the evidence must occur in `issue_resolutions`. Each
resolution has `status` (`resolved`, `accepted_numerical_limitation`, `unresolved`,
`rejected`), `rationale`, and `evidence_refs`. Errors and undefined/unbounded
statistics are explicit issues; they cannot disappear through JSON formatting.
Any `diagnostic_error`, including memory or disk failure, makes generation
ineligible even if a review calls it resolved or accepted. Recollect successful
evidence after fixing resources or requests. A missing calculation is not a
numerical field limitation. Other numerical findings still require scientific
interpretation for the declared study.
Not every diagnostic warning is a physical defect: intentional translational
invariance can produce identical planes, and a material-crossing stencil is not
a vacuum test. Explain the evidence rather than blindly accepting or rejecting.

Optional `external_evidence` lists `{id, path, sha256}` for documented source,
CAD or independent evidence, which the review can cite. Use absolute paths;
changes invalidate the assessment. An absent fact stays unknown. Never create
metadata or a human approval to obtain eligibility.

Any unresolved/rejected category or issue makes generation ineligible. The
library checks evidence identity, source values, reconstruction, support and
implementation. It does not grade the truth of a written rationale or defend
against a hostile local writer. Eligible means a recorded interpretation for
the stated numerical study, not that all physical qualifications are complete.

New generation uses `element_surrogate.generation.v2`. A v1 plan without this
evidence cannot start or resume under the new generator. Preserve old records
as history; do not silently insert an assessment into an old approved plan.
Recollect/interpret/requalify and obtain applicable decisions for a new plan.
Generic benchmarks on existing external labels remain distinct from new field
generation and must disclose their supplied qualification/provenance.
## Unavailable facts are not failed numerical computations

Use [uncertainty and progress](uncertainty_and_progress.md) when facts cannot be
obtained. Unknown geometry, calculation current, internal solver mesh or physical
calibration need not leave a category permanently `unresolved`. After examining
available evidence, a category may be `reviewed` with its unknowns, numerical-only
scope and scientific rationale explicit. Actual field findings can retain
`accepted_numerical_limitation` where that interpretation is defensible. This
does not resolve the physical unknown or declare a vacuum Maxwell violation
physical. Retain raw evidence; do not fabricate metadata or substitute ideal data.

A diagnostic execution/resource error is different: missing computation is not a
measured source limitation. Diagnose/recover, recompute in a supported scope or
return partial results. Keep existing identity, stencil, numerical qualification,
resource and approval checks. The advisory uncertainty tool overrides none of them.

# Simulator-build comparison

When using TRACK, qualify the actual executable, libraries, grid and operating
scope. A process may return zero and complete finite rows while using an
unsuitable compiled grid capacity. `check_track_build_compatibility` compares
caller-supplied independent transverse reference probes in centered straight
laboratory sol3d planes, with at least three step counts and explicit native
component limits. Read status changes, reference disagreements and hash-bound
receipts; successful execution alone is not compatibility. Do not infer a
universal grid limit from a filename or stale log message. Other simulators and
elements use their own adapter/coordinate qualification. This optional check
does not require TRACK for Lorentz-only users or replace analytic controls,
field-specific refinement, field readiness or a physical-field assessment.

