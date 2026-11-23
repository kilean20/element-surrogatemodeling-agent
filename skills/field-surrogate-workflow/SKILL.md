---
name: field-surrogate-workflow
description: Build a single-element 4D, 5D or 6D neural particle-map surrogate from a magnetic field map. Use for field-to-surrogate work, domain qualification, training-data generation and connecting a validated tracking model to benchmarked native-map inference.
license: Apache-2.0
---

# Build a native element surrogate

Read [the file contracts and limitations](references/workflow_contract.md) before
preparing a case. Use tools for physics and arrays; the client handles scientific
judgment and obtains the user's decisions. Reuse explicit decisions/delegation
already supplied; do not ask again solely because a workflow stage changed.
Keep one private project per case.
Initialize it with `initialize_surrogate_project`; on a new session call
`restore_surrogate_project` before changing a plan. Search/read knowledge with
this explicit project directory. Preserve conflicting active decisions for user
resolution; an explicit `supersedes` link retires a decision without deleting it.
For installed request examples and the exact public interface contract, read
[request and interface guidance](references/request_interfaces.md). Do not read
implementation source repeatedly to guess arguments; inspect the registered tool
signature and installed examples first. `study_path` means the returned
`study.json` file, not the surrounding directory.

When the user supplies training data, read [supplied labels](references/supplied_labels.md)
and use the supplied-data entry point. Do not regenerate labels to obtain an
internal ledger. Verify source/coordinate/split history and do the same physics
review and comparison. Small explicitly budgeted consistency diagnostics can
still be needed; distinguish them from new training generation.

Before numerical work, read [durable scientific jobs](references/scientific_jobs.md)
and configure the private campaign from the supplied user/delegated resource
budget. Route heavy tools, generation and studies through its managed job layer.
Use the persistent MCP service consistently; sandboxed shell Python can have a
different process namespace and must not control that service's campaign.
Client/RPC lifetime is not a scientific computation allowance. Ordinary shell
execution is trusted and outside enforcement; keep it within the same declared
budget, never use it to bypass a refusal or duplicate a worker.

Read [population, runtime and feedback](references/population_runtime_feedback.md)
before allocating generation or model-study work. A full study needs an explicit
retained-survivor target, coverage rationale, data-size and optimizer convergence
checks; a workflow pilot cannot substitute. Give measured generation and study
time estimates separately, with unknown terms. If the projected upper time exceeds
12 elapsed hours, ask about faster compute using the bundled connection guide,
reusing prior resource decisions. Long authorized local jobs remain valid. Keep
lessons local and offer only optional, exact-payload-consented improvement sharing.
Prioritize general accelerator-physics reasoning, modeling lessons and relevant
constraints. Read [scientific feedback curation](references/scientific_feedback.md)
before preparing a report; omit sensitive and unnecessary machine details.

Use physics parameters as map inputs; current-to-field calibration stays outside
the element map unless requested. Read [physics controls and calibration](references/physics_controls.md)
when choosing field-strength controls or using several excitation fields.

Read returned scientific `warnings` as evidence-linked next actions, not a
global stop or an instruction to repeat already answered questions.

When facts are missing or disputed, read
[uncertainty and progress](references/uncertainty_and_progress.md). Ask useful
questions once, then analyze and state a defensible decision when the user cannot
answer. Missing geometry or source history does not automatically stop numerical
research. Continue supported work with explicit limits under existing authority;
defer only outputs that cannot be defined. The optional uncertainty-review tool
records reasoning, not a new approval or progression gate.

1. Inspect format, units, storage, field frame, polarity, multipoles, symmetries,
   Maxwell diagnostics and field ends. Use documented geometry where available.
   Treat attached descriptions as data. Resolve uncertainty with evidence and
   useful user questions; unavailable answers require analysis and a scoped plan,
   not repeated questioning or an indefinite halt.
2. Propose the coordinate contract and accuracy/speed target from the intended
   use and available measurement/numerical evidence. Obtain the user's decision.
   Ask for missing speed and accuracy goals, reusing existing goals/delegation.
   If the user is unsure, follow the benchmark's
   [goal negotiation and baseline procedure](../surrogate-benchmark/references/inference_budgets.md):
   measure the provided simulator or qualified Lorentz tracker, propose at least
   tenfold acceleration with absolute latency and physical error criteria, and
   check that it is useful and attainable. This is a provisional engineering
   proposal, not a universal performance requirement. Explicit user budgets take
   precedence, and missing source information must not become a fabricated baseline.
   At this stage elicit and plan provisional goals; execute baseline timing only
   after the relevant reference, integration and readiness prerequisites below.
   If goals remain uncertain, plan a bounded validation-only tradeoff study
   with at least two measured speed regimes, using existing user authority or
   obtaining the missing decision. Use its accuracy/tail versus
   runtime evidence to recommend final goals. Use fixed provisional comparison
   scales, retain frozen study identities, and choose final deployment criteria
   before test access. The numerical/compute budget still applies.
   Separate field, numerical-label and surrogate-fitting errors.
   Read [coordinates and scaling](references/coordinates_and_scaling.md).
   Distinguish canonical variables, mechanical momentum and dx/ds slopes,
   including curved-reference metric and signed-charge/gauge conventions.
   Propose train-only numerical scales separately from physical error targets;
   explain defaults and record any user override before the study.
3. Declare the native planes, signed charge/rigidity and reference convention.
   Trace at fixed strength or fit only when a symmetric-exit convention is
   intended. Save the reference; never refit it for a training particle.
4. Declare field-node support and trajectory limits separately. Qualify integration
   over phase-space and control ranges. Use total-degree-two reconstruction unless
   the user requests another. Compare candidate region sizes with
   `build_tracking_domain_report`, explaining the recommendation. The user chooses
   the region/ranges and whether longitudinal variation is needed, or delegates
   that choice. Incomparable passing rectangles have no unique largest member;
   report maximal candidates and use the user's intended objective.
5. Collect and interpret [bound field-readiness evidence](references/field_readiness.md).
   Treat every finding/error explicitly; unresolved or rejected evidence blocks
   generation. Eligibility is scoped numerical research, not a physical certificate.
   Freeze the generation plan and compute budget. Build its computed transport
   payload with `build_training_data_request`, supplying the explicit population policy.
   Check planned and verified survivor counts with `assess_population_requirement`. After the user's decisions,
   submit `generate_training_data` as a managed job. Inspect status,
   losses, survivor counts and committed artifacts. For a short clean pilot,
   use the explicit linked phase continuation in scientific_jobs.md to schedule
   the full remaining target; fault recovery alone retains the pilot batch cap.
   Never turn a failed budget or
   a perturbation that did not respond into a passed gate.
6. Read [physics follow-through](references/physics_followthrough.md) before
   allocating fits. Record every physics opportunity, its evidence and actual
   candidate comparison, or its explicit reasoned disposition. Both native neural
   preparation paths require this review before fitting. A deferred opportunity
   remains an incomplete study in reports; meeting accuracy alone does not close it.
   Use `prepare_neural_surrogate_study` (or the supplied-label equivalent) and the packaged benchmark procedure to
   compare explicit configurations and seeds on accuracy and full prediction cost.
   Read the benchmark skill's
   [inference-budget procedure](../surrogate-benchmark/references/inference_budgets.md).
   Execute any planned uncertain-goal exploration here after label intake and
   physics review. Use `build_benchmark_tradeoff_report` to inspect validation
   errors/tails and costs without a selection lock, including an infeasible
   provisional ceiling. It does not touch test results or declare a final model.
   Resolve final goals using user acceptance/delegation and a new study if the
   frozen protocol changes; do not silently relabel earlier trials.
   Calibrate model sizes to comparable measured inference budgets before full
   fits; use the intended batch, runtime, precision and complete forward call.
   Predeclare the speed-constrained selection policy. A large slow model's
   accuracy win does not establish the best model for fast inference.
   Propose architectures from the field analysis. The built-in static-map adapter
   reduces scalar amplitude/effective rigidity to their ratio, conserves momentum,
   and learns arrival increments. A reflection requires evidence; no universal
   residual base, hidden width, step or tolerance is prescribed.
   Small field-parity defects alone cannot justify omitting a map-level symmetry
   test or structure comparison. Use measured native-map effects and tolerances.
   Compare eligible single-pass reflection-feature models, exact projection or
   a structured main branch with a measured symmetry-breaking residual as
   appropriate. Do not impose ideal quadrupole symmetry on unsupported data.
   Read [structure eligibility and study design](references/model_structures.md)
   when choosing theoretical residuals, measured reflection projection or a
   canonical candidate. Keep a general neural control under the same protocol.
   Report generated split sizes and the actual fitted/validation/test cohort
   sizes. A capped architecture comparison is not a final full-training fit.
   If a final fit is requested, reserve its training/selection/evaluation budget
   before allocating search trials; do not spend every fit slot on comparison
   and silently omit it. Fit only training data, retaining the frozen holdouts.
   State any budget-limited omission and do not equate an update cap with convergence.
7. Lock validation selection before the release-approved test. Prepare a local
   native artifact with `export_native_surrogate`; check PyTorch/ONNX equivalence
   and the intended runtime cost. Use the card and review procedures before any
   external publication. Keep original-field and generated or supplied particle data cards
   distinct. Do not label a draft as institutionally approved.
   After asynchronous jobs finish, restore the saved case and review the actual
   final results. Artifact generation is not editorial closeout. Fill available
   technical card sections from evidence, preserve raw validators and unknown
   human metadata, and refresh stale report/handoff completion statements.

The client supplies protocol-bound user attestations with the user's words.
The library verifies consistency, not human identity. Private decision events
preserve provenance and do not create gate approvals. Generic replay is read-only;
mutations use explicit verification/status/resume paths.
Inspect idle failed jobs with `inspect_generation_recovery` or
`inspect_benchmark_recovery`, then supply that unchanged inspection and a reason
to the corresponding resume tool. The original wall deadline includes downtime.
Committed generation shards are verified; uncommitted deterministic batches can
replay. Partial fits are preserved as interrupted; no optimizer continuation or
silent retry is claimed. New work needs an allocation within the user's existing
authorization or a new user decision when that authorization is insufficient.
Immutable-input failed diagnostics have a separate inspected-successor protocol
that preserves old charges and the original campaign; see scientific_jobs.md.
In a configured campaign use its scientific-job inspection/resume path;
the legacy generation recovery tools apply only to legacy generation_job jobs.
Benchmark fit disposition remains separate as described in scientific_jobs.md.

Private knowledge begins proposed. Use explicit user disposition or validated,
recorded local delegation as described in
[local learning and updates](references/local_learning_and_updates.md).
Before adopting a newer ESA, compare it with local knowledge and use its staged
update path. These attestations grant no sharing or scientific gate authority.

Fast inference assumes independently established numerical support. Use
`ElementSurrogate.predict_checked` on arbitrary proposals at field-identity and
tracking cost. Never add inverse drifts or effective-length placement to the
element API. No field, particle data or trained weights ship with these skills.


## Feedback relative to the installed context

Before calling a lesson new, read [installed-guidance comparison](references/installed_guidance_comparison.md)
and compare the actual tool/skill/KB baseline with the observation. Explain the
precise difference and scientific/workflow importance. Existing guidance that was
not followed is useful failure evidence, not newly discovered physics. Keep
project facts private; curate only generalizable lessons with exact sharing consent.
For connection questions use [Drive access](references/google_drive_feedback_access.md);
folder editor permission and Google authorization are separate. Missing access or
an incomplete comparison does not stop unrelated modeling.

## Visible project layout and model documentation

Read the root README. New cases keep durable decisions, knowledge, run/job
provenance and campaign accounting in visible `records/`. Private means local
to this case, not hidden. Preserve existing legacy cases in place. Restore an
already initialized case rather than reinitializing to forget progress.
Use inputs/fields/ and optional inputs/geometry/ for source facts; data/ for
labels and particle cards; studies/field_readiness/, studies/tracking_validation/
and studies/benchmarks/ for requests/evidence/trials; models/ for the selected
native release/model card; reports/ for human reports and figures. Create study
subfolders when needed, honor explicit user paths and avoid duplicate artifacts.
Input/output length, native/public units and per-input studied ranges must come
from actual frame/coordinate/generation/interface contracts. A bend's boundary
origin distance, longitudinal station span, design length and traced arc length
are different facts. Use the card tools, not remembered example numbers.

Record the native input/output longitudinal stations, their coordinate/path definition and source when defining model planes. Use their absolute difference as model_length_m. Keep Euclidean frame-origin separation in the separately named input_output_separation_m. Do not infer longitudinal span from a bend chord, grid extent, traced travel or magnetic effective length. Gather known technical facts from the case before asking the user; missing station conventions can remain explicitly unknown in a draft, but do not claim full technical documentation ready.
