---
name: surrogate-benchmark
description: Design and run a bounded accuracy and inference-cost study for a single-element particle-map surrogate. Use when comparing surrogate structures, selecting a model, verifying a model benchmark, or preparing traceable results from 4D, 5D or 6D map predictions.
license: Apache-2.0
---

# Benchmark an element surrogate

Use the tested benchmark tools to compare candidates against one declared map.
Field qualification and coordinate conventions come from the physics tools;
the benchmark measures fitting error against those labels.

Read the workflow's [population and runtime procedure](../field-surrogate-workflow/references/population_runtime_feedback.md)
before allocating fits. Propose a justified full-study survivor target when
delegated, honor explicit minima, inspect representative train-only data-size
learning curves and optimizer histories, and distinguish screening from final
training. Report model-study/benchmark elapsed-time estimates from actual measured
pilots, including preparation, validation, seeds and final fits. Resource choices
above the default 12-hour threshold are a user conversation, not a forced stop.

Read [the protocol and adapter contract](references/protocol.md) when preparing
a study or interpreting its results. Reuse available field/data/reference
records before asking for facts that may already be documented.

1. Define the five benchmark elements: problem and constraints, dataset,
   metrics, reference candidates and reproduction procedure. Propose physical
   candidate structures from the measured field properties. Establish the
   accuracy scales and timing workload from the intended use.
   Ask for missing speed and accuracy goals; reuse explicit goals/delegation.
   When the user is unsure, measure the provided simulator, or the qualified
   bundled Lorentz tracker, and propose at least 10x acceleration with absolute
   times and application/accuracy evidence. Treat this as a proposal requiring
   user acceptance or existing delegation, not a universal success threshold.
   If goals remain uncertain, explore at least two distinguishable measured
   speed regimes within the authorized budget. Compare eligible families at
   similar cost within each regime and show validation error/tails versus time
   before choosing final goals. A third regime is useful when evidence/resources
   justify it. Complexity is a size-calibration knob, not a speed equivalent.
   Read [inference budgets and fair comparison](references/inference_budgets.md).
   Compare accuracy at comparable measured inference cost: calibrate candidate
   widths/depths/features under a common runtime and workload before allocating
   full fits. Keep useful reference maps as speed anchors when available, without
   importing their accuracy claims for a different magnet. Parameter counts,
   equal updates and accuracy-first tie breaking do not provide this comparison.
   Separate physical coordinate conversion, train-only numerical normalization
   and physical loss/error targets. Hold output parameterization fixed across
   base/symmetry ablations, fit scales before base subtraction, and include a
   train-affine control. A whole-recipe ranking alone does not identify causes.
   Read the workflow's [physics follow-through](../field-surrogate-workflow/references/physics_followthrough.md).
   Complete the protocol-bound physics review before fitting: measured map
   symmetry (including useful approximate symmetry), theoretical bases, limiting
   cases, canonical eligibility, controls and scaling. Use actual candidates or
   explicit evidence/rationale for omission. Deferrals are allowed and reported
   as incomplete coverage. A small nonzero field-parity defect is not a map test.
2. Write and validate the versioned protocol with
   `validate_benchmark_protocol`. Declare every coordinate and unit, grouping,
   primary score, seed set, hyperparameter configurations, budget and timing
   scope. Preserve native element planes.
   For a speed-constrained study declare selection policy
   `accuracy_with_latency_budget`, its `max_median_us_per_row` and
   `budget_rationale` before fitting. Declare `latency_bound: exclusive` for a
   strict less-than goal; omitting it retains the older at-most comparison.
   Verify actual public API cost before claiming an end-to-end goal. No complete
   feasible model means no selection; report the unmet budget. An explicit `accuracy_first` policy
   requires explaining why the application values accuracy above latency.
   Report the accuracy/cost frontier and unmatched family coverage.
   Distinguish actual momentum and
   speed from an effective rigidity that represents a field-strength error.
   Explicitly distinguish the generated dataset, a bounded comparison cohort
   and any requested final training phase. State the fitted fraction of the
   training split. Reserve a requested final fit and its honest evaluation
   before spending the search budget. No universal cohort size is prescribed;
   a short study is neither full-data training nor convergence evidence.
3. Obtain the user's target/description, region/range and compute decisions.
   The client records those decisions for the protocol identity; do not write
   approval files on the user's behalf or infer approval from elapsed time.
   These records are local attestations, not identity authentication.
4. `create_benchmark_study` freezes the protocol and checks train/validation
   cohorts. Use `start_benchmark_study` and `benchmark_study_status` to run and
   monitor the declared trusted local training adapter. Inspect failures;
   do not silently add attempts, change budgets or drop difficult rows.
   For uncertain goals, call `build_benchmark_tradeoff_report` after fitting to
   inspect validation observations before any selection lock. It works even
   when the provisional latency ceiling is infeasible, creates no result card
   or selection, and never reads test results. It reports physical component
   errors/tails for every complete configuration, plus the full cost plot.
5. `lock_benchmark_selection` rechecks predictions, weights, units and timing.
   Stochastic seeds are replicates within a configuration; different ridge
   values or hyperparameters are configurations, not extra seeds. Apply the
   declared accuracy/cost rule, then lock one checkpoint before test access.
   Speed-constrained selection retains all required seeds in the configuration
   statistics, then chooses the most accurate latency-feasible seed within the
   winning configuration. Check the actual exported API too. Final goal changes
   need a new declared protocol/study; do not edit a frozen exploratory record.
6. After the user's protocol-bound release decision,
   `evaluate_benchmark_test` scores only that locked model. A fitting change
   after inspecting test results needs a new evaluation plan. The access
   guard is a workflow guard, not filesystem encryption or a code sandbox.
7. `build_benchmark_report` writes a leaderboard, scientific error/cost plots
   and a result-block card draft from saved records. Use its check mode to
   verify results. Explain the declared full forward map, structure rationale,
   measured tails and limitations. Full Genesis card validation and external
   publication are separate steps.
   Check that both speed regimes and dominated candidates remain visible.
   The cost plot shows every complete configuration, with only legend names
   abbreviated; use the full table and hover titles for identities. Explain
   missing regimes/families, timing variation and optimization limits. An
   exploratory protocol's winner is not automatically the deployment choice.

For an independent metric check, `prepare_benchmark_metric_crosscheck` joins
IDs before exporting a bounded cohort. Supply the chosen evaluator's actual
row cap and record its version and observed row count. The export itself is
not an independent evaluation. Setting weights and tails need their own checks.
`score_benchmark_documentation` applies the pinned evidence rubric; its result
is an internal documentation self-assessment.

No previous magnet's region, integration step, rigidity interval, error scale,
tie percentage or model structure becomes a universal default. Keep optional
hosting, cluster and tracking services outside this core procedure. Return
summaries with units and artifact paths; raw particle arrays stay inside tools.

Methodology and two utilities derive from benchmark-builder; see
[attribution](ATTRIBUTION.md). No network fetch or host skill installation is
required to run the packaged tools.
