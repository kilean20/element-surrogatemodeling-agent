# Scientific file contracts

The deterministic APIs are `workflow`, `transport`, `data`, `modeling`,
`benchmark` and `export` in `element_surrogate`. MCP tools wrap the same code.
All artifacts below belong to the user's private project. Arrays never need to
be sent to the language model. The packaged synthetic tests exercise these
contracts without distributing maps, particles or weights.

## Source and reference

During intake, retain available design-length definition/source, field-calculation
current and independent-coil scope, and internal solver-mesh notes. See
the `surrogate-cards` skill's `references/model_card_physical_metadata.md`. The final card
generator computes actual source-grid spacing/extents; it does not infer design
length from the traced-reference travel or missing current from field scale.
Unknown optional source facts remain explicitly not recorded.

`prepare_mapped_field_store` creates a manifest plus exact float64 NPY members.
`mapped_field` is a registered reader. It checks member hashes and reads values
through read-only memory maps. This bounds conversion blocks and subsequent
tracking storage; it does not remove the original text/CST reader's ingestion
cost or accelerate all readiness diagnostics. This is an exact storage change,
not an under-sampled field or internal solver-mesh refinement.

Use `trace_reference_orbit` for fixed field strength and explicit planes, or
`fit_reference_orbit` only when fitting a symmetric exit is intended. Save its
NPZ and hash. Check reference convergence with the same reconstruction and
accuracy needs. A saved reference establishes a convention; it does not by itself
prove the orbit is converged.

A transport JSON contains:

- `schema: element_surrogate.transport.v1`, `native_planes: true`.
- `field`: `path`, `sha256`, optional registered `format` and field `slot`.
- `reference`: `path`, `sha256`, signed `rigidity_tm`, `field_scale`.
- `boundary_convention`: `traced_reference` uses the orbit's orthogonal endpoint
  frames; `declared_native` uses `entry_frame`/`exit_frame` that must equal the
  saved reference's tracing frames. A frame has `origin_m`, `tangent`,
  `horizontal`, `vertical`, finite triples with a right-handed orthonormal basis.
- `codec`: `dimension` = 4, 5 or 6 and `longitudinal` = `time_s`, `ct_m` or
  `path_m`. Six-dimensional states put arrival at index 4 and delta_p/p0 last;
  five-dimensional states put delta_p/p0 last. Four-dimensional states are
  `(x, xp, y, yp)`. Positions are metres; slopes are direction ratios with the
  conventional unit `rad`, not their arctangents or canonical momenta.
- `support` and `trajectory_region`: the common Region specification, with
  `shape`, `radius_m`, optional `half_height_m` for a rectangle, optional
  `z_min_m`/`z_max_m`, optional `bend` using nominal-path arguments. The former
  bounds every accessed grid node, the latter bounds sampled trajectory states.
  These are numerical restrictions unless independent mechanical evidence is
  supplied. The trajectory is checked at numerical stages/endpoints, not proved
  continuously against an unknown surface.
- `reconstruction`: `total-degree-two` by default, explicit `trilinear` or
  `tensor-quadratic` alternatives. No boundary downgrade is hidden.
- `integration`: `method: fixed-rk4` with positive `step_m`, or
  `method: adaptive-rk4` with positive `position_tolerance_m` and
  `direction_tolerance`. Other controls follow the tracking API, including
  plane/path limits and adaptive max-step/turn/retry budgets. Field scale is a
  setting, not an integration-option override.
- For six-dimensional time/ct, `actual_p0_kg_m_s` and `mass_kg` are positive
  actual momentum and rest mass. Effective rigidity may represent a strength
  error and must not supply that speed. For actual momentum variation, the codec
  multiplies both actual momentum and effective rigidity by `1 + delta_p/p0`.

The current reference roll is the upright up-cross-tangent frame; a vertical
tangent is explicitly unsupported. Arbitrary entry/exit frames alone do not
establish a general transported-roll or principal-normal Frenet-Serret solution.

## Qualification and region choice

`qualify_element_transport` takes the transport path, a request and output path.
The request has `ranges` (one lower/upper pair per state coordinate), `settings`
(unique text `id`, signed `effective_rigidity_tm`, `field_scale`), `refinements`
(three or more successively finer complete integration dictionaries),
`random_probes`, `seed`, per-output `component_limits` in native units and
`path_limit_m`. The transport must use the last/fine configuration. Corners,
centre and seeded random probes are tested at every setting; both last adjacent
changes must meet the limits and all statuses must agree. No surviving common
probes means inconclusive, not success. Numerical qualification is empirical.

An optional `effective_coupling_range` explicitly authorizes interpolation in
scalar amplitude/effective rigidity. Its endpoints must equal the tested
settings' extreme ratios. Without it, inference accepts only the listed
settings. This declaration is bound to qualification and generation approvals;
finite probes do not become a continuum error theorem.

Use the same probes/ranges/criteria for candidate-size reports. The largest pass
is a numerical recommendation, with survivor counts, not an inferred bore. The
user chooses and decides whether a longitudinally varying region is needed.
Read Maxwell/material evidence separately. `SupportedField` rejects any query
whose actual reconstruction nodes leave support; the index selectors are shared
with sampling. `check_track_domain_isolation` supplies optional black-box
perturbation/sentinel/loss-timing evidence with caller-supplied runtime and probes.

## Generation and sealed cohorts

A generation plan has `schema: element_surrogate.generation.v2`, the exact
`transport` description, `qualification_sha256` (canonical JSON identity),
`ranges`, `settings`, optional `effective_coupling_range`, `seed`, `batch_size`,
`max_proposals_per_setting`, `max_wall_seconds`, and a nonempty
`material_boundary_evidence` statement (explicitly unknown is permitted for a
numerically restricted research map).
The plan also embeds `field_readiness` and its canonical
`field_readiness_sha256`; see [the evidence/review contract](field_readiness.md).
Each setting adds `split` = train, validation or test, and requested
`surviving_rows`. Distinct groups must have
distinct effective couplings; renaming identical physics does not hold out a
setting. The field shape is fixed; amplitude scaling is explicit.

`validate_training_data_plan` checks identity before work. The client supplies
the user's three protocol-bound description/target, region/range and compute
attestations in the benchmark approval schema, using the generation plan's
canonical identity. No tool creates those records or authenticates the speaker.

`start_training_data_generation` starts a hidden local supervisor. It kills and
waits for its direct child at the approved wall deadline. `training_data_status`
reports progress and verifies committed hashes. Proposals and survivors are
separate immutable shards; losses, IDs, path lengths and unused surviving
proposals remain recorded. Resume uses deterministic proposal streams and
verifies orphan/committed bytes. Failed jobs require inspection. Source and
numerical identities are checked before work and at controlled
completion/pause/budget boundaries. A detected change invalidates the campaign;
it cannot be resumed or exported as complete. Inputs must remain stable during
execution; these checks are not a security sandbox against hostile concurrent writes.
The Python
`data.generate` API can pause at a bounded batch boundary; its own soft deadline
is checked between batches, so use the supervisor for a hard wall limit.

`prepare_training_benchmark_cohorts` writes bounded JSON descriptors that pin
member hashes and prefix counts, balanced across settings. It opens no test
particle arrays. This is a disclosed capped cohort, not all generated rows.
The benchmark reader materializes only the requested, approved split with a
decompression/concatenation guard. This is not a process-wide RAM limit.

## Neural study and export

`prepare_neural_surrogate_study` takes a request with `name`,
`maximum_rows_per_split`, per-coordinate `error_scales`, `access_and_reuse`,
`software`, and explicit benchmark `candidates`, `selection`, `timing`,
`resources`. It produces the existing benchmark protocol; it starts no training
and creates no approvals. Its adapter source and scientific identities are pinned.

Each built-in candidate configuration declares:
`schema: element_surrogate.static_magnetic_mlp.v1`, `dimension`, nonempty
`hidden_widths`, `base` = none or train_affine, `dtype` = float32 or float64,
`device`, `threads`, training `batch_size`, `inference_batch_size`,
`learning_rate`, and positive dimensional `loss_scales`. Optional
`reflection_group` is a closed sign group on transverse coordinates, including
identity, with `symmetry_evidence_sha256` required for a nontrivial group.
The implementation generates the actual forward description; a caller's
architectural explanation is retained separately. No conserved-energy head is
trained. The network sees amplitude/effective rigidity once and excludes absolute
arrival, then adds the arrival increment and passes momentum through in inference
precision. The affine base is fitted only on training rows, not an ideal magnet.
Seeds/configurations, step/wall budgets, timing and selection are the benchmark
contract; initial checkpoints are eligible. PyTorch is optional and must exist
in the MCP/worker environment. A non-scaling excitation family needs a different
backend; do not silently treat saturation as an amplitude factor.

Follow `surrogate-benchmark` for validation selection and release-approved test
access. The held-out test cannot choose the architecture. Optional ONNX/ONNX
Runtime export verifies full native-map output at several batch sizes on
validation, copies portable CPU weights, and writes a versioned contract. Before
export, portable CPU predictions must also agree with the selected checkpoint's
saved validation predictions, joined by exact IDs. Declared input precision is
preserved separately from arithmetic precision. ONNX embeds the native units,
coordinate definitions, plane convention and physics identities as metadata.
Torch/ONNX are runtime requirements for this operation, not bundled assets.
Their installed versions are recorded. ONNX equivalence does not transfer a
PyTorch speed claim to the deployed runtime; benchmark that path separately.

`ElementSurrogate` loads a verified local artifact, backend torch or onnx. Inputs
use native SI values plus effective rigidity and amplitude; control values must
match the declared settings/range. `predict` requires an explicit declaration
that numerical support was established independently. `predict_checked` also
verifies the transport identity and tracks proposals; rejected proposals have
NaN predictions and a status, with -1 reserved for outside the declared scope.
Neither API wraps inverse drifts or alters the element's placement.

Use the packaged card and review skills for complete Genesis documentation and
retained validator receipts. Field and particle datasets need separate cards.
Human metadata and institutional decisions cannot be inferred from a checkpoint.
Local artifact preparation performs no registration, upload or remote publication.
