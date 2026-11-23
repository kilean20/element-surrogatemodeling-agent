# Cohorts, documentation and optional simulator compatibility

These tools support the existing field-to-model workflow. They do not choose a
physical aperture, change numerical tracking defaults or create release rights.

## Training-only scale proposals

`propose_training_coordinate_scales` accepts a numeric NPZ cohort or the JSON
shard descriptor produced by `prepare_training_benchmark_cohorts`. Descriptors
must explicitly declare `split: train`. Supply `protocol_path` to bind the
training path, hash, metadata and row count to the frozen benchmark protocol.
Legacy NPZ callers remain supported; without a protocol, their native-unit and
training-split designation is a caller contract, not independently established.

The helper uses every declared row. It opens one NPZ shard at a time and merges
batched, magnitude-scaled sums of squares. Its `working_memory_bytes` allowance
is checked against decompressed shard size before loading. An oversize shard
gets an actionable refusal to reshard or increase the allowance; it is never
silently subsampled. This bounds working arrays, not process RSS. The declared
cohort may itself have an explicit benchmark row cap; that cap is reported by
the cohort and is not changed by this helper.

The output contains the row count, input identity, execution allowance, raw
target RMS, zero-component fallbacks and common transverse magnitude proposals.
6D arrival uses the output-minus-input increment and conserved momentum is not
learned. Numerical scaling remains separate from physical error tolerances and
from eligibility for canonical/symplectic modeling. See the packaged coordinate
and scaling guidance for the slope-to-mechanical-momentum relation.

## Model and dataset cards

Supply `model.sections` using pinned template headings. Case, repeated whitespace
and trailing periods are normalized consistently. Unknown or colliding keys are
refused before card output. Metadata review and card results include section
coverage. The license section remains derived from that artifact's explicit
rights and is identified as a controlled section rather than silently ignored.

For supported native checkpoints, the model description includes the actual
fitted base, network parameter count, exact zero-output-layer test and selected
optimizer step. A zero neural correction does not make a fitted base untrained;
a step-zero winner does not establish neural improvement or convergence.
Other adapters remain explicitly uninspected. This does not force a model family
or replace common validation selection with another rule.

Genesis assets and strict validators stay pinned. Their optional software
`version` slot accepts only a numeric version, while a local `identifier` can
preserve an exact prerelease. When this generator supplies its own creator
identity it records `element-surrogate@<exact-version>` and omits the optional
numeric slot for a prerelease. It does not pretend the stable release exists.
Supplied creator records are not silently rewritten. Metadata preflight reports
incompatible supplied versions with an explicit suggested identity, and raw
validator findings remain available.

Missing owner/contact, original authors, sponsor/research organizations,
sensitivity and publisher facts remain separate from software-schema findings.
Local draft validity, full upstream conformance and release authority are
different statuses. Default software provenance supplies no dataset authorship,
classification, license, publisher or institutional approval.

## Optional TRACK build comparison

`check_track_build_compatibility(request_path, output_dir)` tests a specific
runtime, field table and supplied independent reference probes. Use the installed
`track-compatibility` request example. It requires hashes, native m/rad plane
direction-ratio states, reference provenance, momentum constants, at least three
increasing step counts and separate refinement/compatibility component limits.

The current adapter is for transverse `sol3d` comparisons in centered straight
laboratory planes. It does not qualify bend-reference transformations or 6D
arrival coordinates. Another simulator/profile can use its own tested adapter;
TRACK is not an ESA dependency. Reference independence is a caller attestation,
not something a file hash authenticates.

The receipt retains actual grid dimensions, executable/library/field/probe hashes,
all refinement outputs, status changes and differences from the supplied
reference. A successful process exit cannot by itself pass compatibility. A
failure on one grid does not establish a universal size limit: compiled builds
and stale diagnostic strings can differ. Repeat the comparison after changing
the build, field, reconstruction, frame or operating range. Analytic controls,
per-field convergence and field readiness remain separate evidence; agreement
with TRACK alone neither proves physics nor determines the most accurate
continuous-field reconstruction. The receipt grants no generation/release approval.
