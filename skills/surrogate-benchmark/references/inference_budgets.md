# Accuracy at a measured inference budget

A useful particle-map surrogate must satisfy its intended accuracy and speed.
An unconstrained accuracy leaderboard followed by speed tie breaking answers a
different question from best accuracy at comparable inference time. No family
has an automatic right to more runtime or a guaranteed accuracy advantage.

## Establish the application workload

Ask the user for the intended use and missing speed/accuracy goals before
allocating expensive fits. Reuse goals, constraints and delegation already
recorded; do not ask again just because a workflow stage changed. Obtain or
propose both an absolute latency/throughput budget and per-coordinate accuracy
criteria, including units, RMSE and relevant tails. Numerical normalization and
loss scales are not themselves acceptance criteria. Mark proposals and delegated
choices explicitly, with their rationale, before freezing the protocol.

Obtain or propose batch sizes from intended use: single-ray optimization and
large particle ensembles can have different rankings. Specify hardware, threads,
dtype, runtime/backend, transfer/synchronization, API scope, and included input
processing, theoretical base, symmetry features/projection, output conversion
and bounds checks. Keep loading and trajectory support checking separately
priced. Compare complete calls with complete calls, and tensor cores with tensor
cores; do not subtract wrapper overhead from one model alone.

If an existing published map is a useful speed reference, measure it under that
same workload. A different magnet is only a latency anchor. State any difference
in native precision/API scope. A common-precision diagnostic can help attribute
cost but does not qualify a cast deployment or change its measured error.

## When the user is unsure

Use a short, explicitly budgeted pilot to measure the user's provided simulator.
If no simulator was provided, use this package's qualified Lorentz tracker when
the required source field/reference information is available. Establish numerical
adequacy over the intended region/controls first: step size and field reading
must support the proposed accuracy. A simulator agreement check alone is not
proof of absolute field accuracy. Do not change numerical settings to distort
the speed comparison.

Propose **at least a tenfold speedup** as an initial engineering goal:

`speedup = measured_reference_time / measured_surrogate_time >= 10`.

Equivalently, propose a surrogate latency no greater than the measured reference
latency divided by ten at the same workload. Show absolute times too. This is
a usefulness heuristic, not a physics requirement or evidence that the proposed
goal is sufficient. Compare it with the intended application and inexpensive
surrogate prototypes/reference maps: an expensive tracker can yield an overly
loose ceiling, while an already fast simulator can make tenfold acceleration
unattainable. Recommend a practical goal or disclose infeasibility with evidence.
Retain faster competitive models on the accuracy/cost frontier.

Match native planes/reference, requested outputs, input/control distribution,
batch size, hardware and thread/resource allocation, and warm/cold scope. State
precision differences and verify accuracy before treating them as a valid
engineering configuration. Include transfers/synchronization and required API
processing on both sides, and report startup/field-loading/I/O or support checks
separately when excluded. Use a common supported cohort and disclose losses;
state prediction on independently supported inputs and checked prediction on
arbitrary proposals are distinct workloads. Benchmark the one the application
needs, including per-trajectory checks when required. In either workload,
early loss must not masquerade as full-map acceleration. Retain repeated timings
and their variation. A noisy value close to ten is borderline, not a robust
speedup claim. Freeze the accepted target before model selection/test access.

For supplied labels with no executable simulator or accessible source field,
report that a measured baseline is unavailable. Use traceable compatible timing
records or propose an absolute goal from the application/prototypes; ask for
missing information when needed. Do not fabricate a speedup or regenerate a
large training dataset merely to time the source.

Propose accuracy from the user's beam/measurement tolerances, native coordinate
units and numerical-label evidence. Separate surrogate fit errors from source
field uncertainty and transport-discretization sensitivity. A deterministic map
can be fitted more closely than its physical uncertainty; that does not improve
field truth. Report provisional targets where application evidence is absent,
and use user acceptance or existing delegation to proceed. No universal position,
angle, energy or arrival-error threshold is prescribed.

An explicit user speed goal takes precedence over this unsure-user fallback.
Do not silently add a tenfold requirement to an already supplied absolute budget.
Combine criteria only when the user accepts or delegates that choice. Likewise,
do not relax an explicit strict ceiling when it proves difficult to meet.

## Explore the tradeoff before choosing uncertain goals

After the declared fits, use `build_benchmark_tradeoff_report(path, output_dir)`
(Python: `build_tradeoff_report(study_path, output_dir)`) to inspect validation
evidence without a selection lock. It validates complete trials, shows all
configurations even when the provisional ceiling is infeasible, and neither
opens test data/results nor creates a selection or result-card draft. Use a
separate output directory from the selected-model report. The declared
incomplete-configuration policy still applies.

When speed or accuracy goals remain uncertain, plan a bounded validation-only
exploration with **at least two meaningfully different measured speed regimes**.
Use cheap prototype timings to choose a faster regime and a slower, higher-cost
regime relevant to the application. Complexity/width/depth/features are knobs
used to reach them; parameter count is not an equivalent speed measurement.
Choose separation that is distinguishable given timing variation, not a
universal speed ratio or network size. Add an intermediate regime when the
initial comparison suggests a useful transition and the compute budget permits.
Two regimes provide a minimum tradeoff comparison, not a smooth curve or a
well-established knee.

Compare eligible model families within each regime, using measured sizes near
comparable inference budgets. Keep common data/splits, native units, train-only
normalization, loss/metric definitions and comparable tuning opportunities, with
declared seeds and resource limits. Disclose families that cannot reach a regime
because of their minimum runtime or the available fit budget; do not force
equality, omit the reason, or fabricate coverage. An authorized comparison budget
that cannot support two regimes is an explicit incomplete exploration to resolve
with the user, not silent permission to add fits.

Use fixed provisional comparison scales when final acceptance goals are unknown;
label them as such. Show validation component RMSE and relevant tails in physical
units, the fixed aggregate score, measured time/variation and the Pareto frontier.
Present faster and slower candidates even when their accuracy is worse. A slower
model may be dominated, a plateau may occur, or differences may be inconclusive.
Limited optimization is not evidence of the best attainable family accuracy.

Use this evidence to recommend goals or a useful tradeoff, then obtain the user's
choice or use existing delegation. Preserve exploratory protocol/selection locks.
Changing a frozen ceiling, metric or selection rule requires a new declared study;
reuse prior artifacts only through supported, verified paths retaining their
identity/history, never by relabelling old trial records. Freeze the final
deployment selection before test access. Exploratory test inspection is not part
of selecting budgets; previously used test data retain that history.

An explicit deployment ceiling stays binding. Exploration can include a faster
regime and one near that ceiling. Above-ceiling references, if within the approved
research scope/resources, are labelled ineligible for deployment; their presence
does not relax the goal. No universal microsecond or complexity tiers ship.

Changing goals is not an automatic trial migration. The existing trusted-adapter
contract can reuse verified frozen checkpoints in a new study: bind the parent
checkpoint/configuration/seed hashes and source history, load the actual model,
save unchanged model bytes and use `max_steps: 0` for that frozen evaluation.
Recompute validation predictions and timings under the new protocol; do not
relabel old prediction/trial files. Verify native units, controls, planes, data
and model function, including parent golden outputs when software differs.
Record transferred training and prior test use, and charge the new validation/
timing work to its authorized budget. This is a caller-defined adapter route,
not an automatic reuse feature of native study preparation. Retraining remains
appropriate when the intended model/data/objective changes.

## Calibrate before expensive fitting

1. Propose latency budgets from the application/reference measurement, with a
   rationale. There is no universal microsecond budget, matching ratio, hidden
   width or batch size. Use existing user delegation; otherwise ask for the
   application decision as needed.
2. Time cheap initialized prototypes at candidate widths/depths/feature counts.
   Normalization and any fitted base use training data only. Keep a shared
   dtype/backend for structural ablations; if precision/runtime changes are
   explored, treat them as explicit configurations and verify numerical error.
3. For each family, find sizes near the same budget or within a declared latency
   band. Retain smaller faster controls and the observed Pareto frontier. A
   budget ceiling should not exclude an equally accurate faster model merely
   to force equality. Disclose a family that cannot be brought into the band.
4. Freeze configurations, seeds, matched train/validation rows, physical loss and
   train-only scaling. Use comparable tuning opportunities; measure learning
   curves. Equal update counts alone do not establish optimization convergence.
   Keep base/symmetry ablations separate from whole-recipe engineering rankings.
5. Re-time restored fitted checkpoints using warmups and repeated measurements;
   retain raw repeats and dispersion. Interleave or repeat near-budget timings
   to assess drift. A median within measurement variation of a ceiling is a
   borderline observation, not a hard real-time guarantee. Freeze a new protocol
   for an altered budget; do not silently rewrite a locked study.

## Enforce and report the selection

For a speed-constrained study set:

```json
{
  "policy": "accuracy_with_latency_budget",
  "primary": "worst_scaled_setting_mean_rmse",
  "relative_accuracy_tie": 0.02,
  "incomplete_policy": "refuse",
  "max_median_us_per_row": 3.0,
  "latency_bound": "exclusive",
  "budget_rationale": "Example application budget at its declared batch and CPU"
}
```

The numbers are illustrative, not defaults. `latency_bound: exclusive` means
strictly less than the ceiling; `inclusive` means at most the ceiling and is
the legacy-compatible default when the key is omitted. The tool first filters complete
configurations by the declared cost ceiling, then applies the accuracy/tie rule
within those configurations. No feasible configuration means selection refuses.
Configuration cost is the median across required seeds of each trial's median
per-row time. All required seeds remain in the accuracy/cost statistics. For
deployment selection, the configuration cost must pass and the most accurate
measured latency-feasible seed supplies its checkpoint. Its own median must
also pass; a faster aggregate cannot excuse an over-budget checkpoint. This
does not constrain P99 latency, worst-case latency or every seed. The report
retains all complete configurations, the observed nondominated accuracy/cost
frontier and the latency-eligible list. Compare per-coordinate errors and tails
as well as the dimensionless primary score; meeting a latency ceiling alone
does not mean accuracy targets passed.

The validation comparison JSON/table includes physical mean seed RMSE,
setting-mean RMSE, mean seed P99 and worst seed P99 for every complete
configuration. Mean seed quantiles are not pooled quantiles or uncertainty
bounds. The selected-model report separately reports the chosen checkpoint's
validation or final-test errors; its validation comparison never becomes a
test-based configuration ranking.

The timing constraint applies to the declared adapter scope. Before release,
also measure the actual intended public API with all required wrappers,
conversions, bounds and transfers. An adapter-only pass does not establish a
full-API goal. If the scopes differ during search, measure their overhead and
reserve evidence-based headroom, then verify the final call; no fixed overhead
percentage is prescribed. Report both measurements and any unmet goal.

`accuracy_first` is an explicit alternative when accuracy is the application
priority. Historical protocols without `policy` retain their original selection
and record shape. They must not be described as matched-inference studies.

A new study may evaluate several predeclared budgets for scientific comparison;
one deployment budget and validation selection still need to be frozen before
test access. A post-hoc frontier from already studied data is exploratory, not
a fresh held-out or independent native-agent acceptance result. If no comparable
family/tuning coverage exists, report the open comparison rather than asserting
that physics structure is ineffective or that a slow model is best for use.
