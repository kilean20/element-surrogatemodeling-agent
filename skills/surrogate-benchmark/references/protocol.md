# Protocol and adapter contract

The public Python API is `element_surrogate.benchmark`; the MCP tools use the
same functions and write ordinary run records. This layer can train caller-defined
models through an adapter. The field-surrogate workflow also provides bounded
generation and static-magnetic neural adapters; additional field families and
architecture classes need their own adapters and qualification.

## Protocol

Use schema `element_surrogate.benchmark.v1`. Required records are:

| Key | Meaning |
|---|---|
| `name`, `coordinate_convention` | Study identity and entrance/exit coordinate definitions |
| `coordinates` | Four, five or six output records: `name`, `unit`, `definition`, `role`, positive `error_scale` |
| `inputs`, `input_units`, `forbidden_inputs` | Exact model-input names/units and excluded derived/label columns |
| `physics` | SHA256s of field, reference, reconstruction, integration qualification and numerical region; `native_planes: true`; `qualification_scope` and `boundary_convention` |
| `dataset` | `identity_sha256`, `access_and_reuse`, and `splits` for train/validation/test |
| `candidates` | Configuration records, listed below |
| `selection` | Primary score, nonnegative `relative_accuracy_tie`, `incomplete_policy`, explicit `policy`; a speed-constrained policy also needs `max_median_us_per_row` and `budget_rationale` |
| `timing` | Full adapter-call timing contract, listed below |
| `resources` | Explicit positive row, array-memory, checkpoint and wall-time budgets |
| `reproduction` | Trusted `adapter_path`, its `adapter_sha256`, and software-environment description |

`dataset.split_unit` defaults to `setting`: physical setting groups must be
disjoint across splits. An explicit `particle` value permits shared physical
settings while retaining unique samples and train/validation/test sample-ID
overlap checks. Reports disclose this weaker generalization question. Never
rename a shared setting to make it look held out. Test sample overlap is checked
after release-approved access; the exporter must establish split membership
without exposing test labels during fitting.

The first four roles are position/slope/position/slope. A fifth coordinate has
role `energy`; six dimensions add both an `arrival` and an `energy` role in
the declared order. Roles specify how to read a coordinate, not a canonical
symplectic form. The benchmark does not convert units or prove physical timing:
the source coordinate contract owns those calculations. Errors stay in the
declared units and are divided by their own error scales for the primary score.

Each split record contains `path` (relative to the protocol file), `sha256`,
`rows`, and nonempty distinct text `groups`. Groups must be disjoint across
splits. A dataset identity is a manifest/specification identity, while the split
SHA256s identify actual file bytes. Protocol validation checks declarations;
creation checks actual train/validation data and source hashes. Test file contents
are not read during creation or fitting.

Resources are `max_rows_per_split`, `max_npz_uncompressed_bytes`,
`max_checkpoint_bytes`, `max_study_wall_seconds` for the fitting campaign, and
`max_test_wall_seconds` for final evaluation. Cohorts are explicitly bounded and
held in memory for exact empirical tails; this is not a streaming large-dataset
backend. Each candidate also has its own worker deadline and step budget.

## Cohorts and predictions

A cohort NPZ contains numeric `inputs` with shape `(rows, len(inputs))`, numeric
`targets` with shape `(rows, dimension)`, one-dimensional integer or text
`sample_id` and `setting_id`, and metadata `input_names`, `input_units`, `coordinate_names`,
`coordinate_units`, `dataset_identity_sha256`. Do not use object arrays/pickle.
IDs must be unique within each split; setting membership must match the declared
groups. Internal cohort order is canonical text-ID order; every row-aligned
array follows that order. Train/validation ID overlap is checked at creation; test overlap is
checked only after authorized test access.

Prediction NPZs contain `sample_id`, `prediction`, scalar `split`, scalar
`protocol_sha256`, scalar `checkpoint_sha256`, scalar
`dataset_identity_sha256`, plus `coordinate_names` and `coordinate_units`.
Use `scoring.write_predictions` to construct them. Scoring joins IDs before
computing errors; missing, duplicate, extra, nonfinite and malformed predictions
are refused. The public `score_predictions` API only accepts train/validation;
use `evaluate_test` for guarded final evaluation. Internal readers are not an
OS security boundary against trusted code directly accessing the filesystem.

Global component RMSE is the square root of the mean squared component error.
`setting_mean_rmse` computes component RMSE within each setting and then averages
those setting RMSEs equally. These are different statistics. Primary score is
the largest component RMSE divided by that coordinate's error scale, using
either `worst_scaled_rmse` or `worst_scaled_setting_mean_rmse` as declared.
Absolute P95/P99/P99.9 use NumPy's empirical `inverted_cdf` convention. Maxima
describe the evaluated cohort, not a continuous-domain guarantee.

## Candidates and selection

For uncertain goals, use the [tradeoff procedure](inference_budgets.md) to design
at least two distinguishable measured speed regimes. Arbitrary configurations
already support these experiments; regime coverage is a study-design obligation,
not inferred from different parameter counts or guaranteed by the protocol
validator. The report plots every complete configuration and retains the full
table. Exploratory validation comparisons do not change a frozen deployment
selection rule or authorize test-based goal tuning.

Each candidate declares `id`, `architecture`, `kind` (`stochastic` or
`deterministic`), distinct integer `seeds`, a JSON `configuration`, `rationale`,
`forward_description`, `max_steps` and `max_wall_seconds`. Configurations within
one architecture have matched seed sets. A deterministic fit has one seed;
different regularization values are separate configurations.

Each configuration is scored by its mean validation primary score across seeds.
For `policy: accuracy_with_latency_budget`, first retain only configurations
whose median inference cost satisfies `max_median_us_per_row`. Optional
`latency_bound` is `exclusive` for a strict less-than ceiling or `inclusive`
for at most it; omitting the key retains the inclusive comparison. No complete
feasible configuration means selection refuses. The cost is the median of
seed-level median timings, not a tail/worst-case bound. The `budget_rationale`
states the application basis; see [inference budgets](inference_budgets.md).
For explicit `accuracy_first`, all complete configurations enter the accuracy
comparison. Historical protocols omitting policy keep that original behavior
and record shape. New explicit-policy selections also retain the complete
accuracy/cost Pareto frontier and latency-eligible candidate list.

Within the eligible cost set,
All complete configurations within the declared relative tie of the smallest
mean score are eligible; lowest median inference cost wins. The best-validation
latency-feasible seed within that configuration supplies the selected checkpoint
for the speed-constrained policy. Its own recorded median must satisfy the same
bound, while every declared seed remains in configuration statistics. Accuracy-first
policies select the best-validation seed without a latency filter. Exact remaining
ties use score/ID/seed ordering as recorded in the implementation. Per-architecture
summaries are informational: they do not dilute a good configuration by averaging
it with poor hyperparameters.

`incomplete_policy` is `refuse` or `exclude_configuration`. The latter excludes
the whole configuration if any required seed fails; failures remain recorded.
No automatic additional seed or trial is launched. Completed trials can be
reused unchanged, and selection is immutable once written. An interrupted worker
still marked running requires inspection; the runner refuses to launch a duplicate.

## Trusted adapter

Before locking a final deployment selection, `build_benchmark_tradeoff_report`
can validate/report all complete configurations on validation, even with an
infeasible ceiling. It creates no lock/card and reads no test files/results.
Its per-configuration physical errors/tails are means/maxima of seed statistics,
not pooled quantiles. The selected-model report remains a separate post-lock
path; final goal changes do not mutate existing protocols or trial identities.

Supply a Python file defining:

```python
def build_adapter(
    train_inputs, train_targets, configuration, seed
): ...  # return a model with step(), predict(inputs), save(path)


def load_adapter(
    checkpoint_path, configuration
): ...  # restore the complete inference model; predict(inputs) is required
```

One `step()` is one declared optimization unit. The runner performs at most
`max_steps`, measures validation after each step, saves the best-validation
checkpoint and checks restored predictions before timing. The factory receives
only approved inference columns and training targets. All preprocessing,
non-trainable parameters, bases and decoders required by `predict` belong to the
saved/restored inference model. `save` must write one file at the exact supplied
path, without appending a suffix. Adapter code is trusted local executable code;
the child process imposes a deadline, not a security sandbox. It must not spawn
unmanaged child jobs. The caller's environment must already supply its framework.

Timing requires `batch_size`, `warmup_calls`, `repetitions`, `threads`,
`input_dtype` (`float32` or `float64`), `scope: adapter_predict_call`, `hardware`,
`device`, `included_operations`, `excluded_operations`, `golden_atol` for each
coordinate, and `synchronization`. Use `synchronous_cpu` only for CPU; otherwise
use `adapter` and implement `model.synchronize()`. Timings cover the adapter's
full input-to-output call and required synchronization, excluding the runner's
post-call correctness checks. Input/output transfers inside that call are included.
Raw repetitions, median, P10/P90 and measured environment versions are saved.

## Approval and execution

Approval JSON has schema `element_surrogate.approval.v1`, the exact canonical
`protocol_sha256`, `gate`, `decision: approved`, nonempty `actor` and verbatim
`user_statement`. Required fitting gates are `description_target`, `region_ranges`
and `compute_budget`; `release` permits final test evaluation. The client must
obtain and record the user's decision. These files bind an attested decision to
a protocol but do not authenticate the actor. There is no auto-approval tool.

Python flow:

```python
from element_surrogate.benchmark import create_study, run_study, lock_selection

# Paths and user-attested records are supplied by the current project.
created = create_study(protocol_path, new_study_directory, approval_paths)
run_study(created["study_path"])
locked = lock_selection(created["study_path"])
```

The corresponding MCP background path uses `create_benchmark_study`,
`start_benchmark_study`, `benchmark_study_status`, then
`lock_benchmark_selection`. Starting twice does not duplicate work. The worker
deadline covers initialization, fitting, checkpoint checks and timing. The
coordinator's overall fitting deadline includes idle time between resumptions.

## Reporting and independent checks

`build_benchmark_report` writes full raw-unit tables and normalized component
plots, a validation accuracy/cost plot, source identities and a result-block
card draft. Existing evidence is checked rather than silently overwritten;
use a new output directory after adding authorized test results. Its check mode
detects changed sources/units/identities and edited result blocks. The two pinned
upstream utilities and their license/rubric total less than 40 KiB before local
manifest/notice files. No upstream example datasets or weights are bundled.

`prepare_benchmark_metric_crosscheck` exports an evenly spaced sorted-ID cohort,
at most `maximum_rows`, with exact counts, units and CSV hash. The caller must
choose the external evaluator's actual row cap and check the observed count.
The export records that no evaluator has yet run. An independent MAE/RMSE check
does not validate setting balancing, tails, field accuracy or timing physics.

Full Genesis model/data cards, registry submission and external publication are
not implied by a generated result block or a documentation rubric score.

Generic run-record replay refuses mutating tools. Use status/check/lock/evaluate
on existing artifacts; fresh-directory campaign replay orchestration is later work.

## Optional train-only polynomial residual base

The static-magnetic neural adapter accepts `base: train_polynomial` with explicit
positive integer `base_degree` and `max_base_features`. It fits all total-degree
monomials, including a constant, on training-normalized inputs only, then learns
a tanh residual. The feature budget is checked before allocating the design.
It does not cap the whole process memory; training rows and inference batch
limits still matter. Arrival is excluded from features, momentum passes through,
and the declared reflection projection acts on the complete map as before.

Use this candidate to test whether a low-order base simplifies learning a smooth
map. Degree is a validation-selected study choice, not a universal default or a
field interpolation order. It asserts neither a Maxwell eigenmode nor canonical
symplecticity. Its additional coefficients and complete prediction cost belong
in the comparison. A fitted base may win at optimizer step zero. Configuration
and base buffers are saved/reloaded together; never fit the base on validation
or test targets.


A candidate's max_steps is a nonnegative integer. Zero is a legitimate fixed
baseline: initialization or train-only base fitting, initial validation, saving,
reload equivalence and full-call timing still run and count against resource
budgets. No optimizer update runs. Report step zero explicitly; do not approximate
it with a tiny learning rate. Positive wall/memory/data budgets are still required.
