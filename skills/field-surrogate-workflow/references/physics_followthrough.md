# From a physical hypothesis to a model comparison

Read this before allocating fits. The goal is to exploit useful physics without
silently replacing a realistic field by an ideal magnet. A small numerical parity
defect neither proves nor disproves that an approximate map symmetry is useful.
Conversely, magnet type alone cannot justify imposing an exact symmetry.

## Required record, flexible scientific choices

Both generated and supplied native neural protocols require a physics review
before `create_benchmark_study`. Supply `physics_review` with the preparation
request, or call `review_model_physics(protocol_path, request_path, output_dir)`
and use its returned protocol. Record all seven topics: controls,
reference_and_coordinates, symmetry, theoretical_base, limiting_cases,
canonical_structure, scaling_and_metrics. Use actual candidate IDs, rationales
and file/SHA256 evidence. `element-surrogate request-example physics-review`
prints the format. No tool supplies a user's approval.

Use `compared` for an actual declared comparison, `established_contract` for a
bound source/coordinate contract, `measured_not_supported` for measured
ineligibility, `not_applicable` with its evidence, or `deferred` with its reason.
Deferrals permit fitting; the report retains incomplete coverage. A small
budget can justify a scoped comparison. It cannot justify claiming that every
important structure was explored. This record checks design consistency, not
the truth of arbitrary narrative. Review actual outcomes at closeout.

## Reflection features: one network evaluation

For independent horizontal and vertical pair reflections, define
`a=(x,xp)`, `b=(y,yp)` in coordinates whose physical center is zero. The inputs
to an invariant coefficient network can be

```
x*x, x*xp, xp*xp, y*y, y*yp, yp*yp, coupling
```

where every coordinate is first divided by its own positive training RMS and
`coupling=field_scale/effective_rigidity_tm`. For 5D/6D, momentum deviation is
an additional invariant; absolute arrival is omitted and its increment learned.
Horizontal output corrections are `x*f(I)+xp*g(I)`; vertical corrections are
`y*h(I)+yp*j(I)`. This is a character decomposition of the reflection group.
It enforces the stated pair parities with one coefficient-network call.
It is neither a Maxwell eigenmode expansion nor a symplectic construction.

Use `model_family: reflection_features`, `input_normalization:
zero_centered_rms`, a closed `reflection_group` and the measured
`symmetry_assessment`/`symmetry_evidence_sha256`. Independent pair reflections,
one-pair reflection and central inversion have different characters. The tool
derives their permitted features/gates rather than assuming they are identical.
It also constrains any fitted affine/polynomial base; an unrestricted base would
destroy the guarantee. Feature products are dimensionless and normalized using
training only. The physical origin is never replaced by an empirical mean.

Use `assess_map_symmetry` across the real input/control envelope, native planes,
reference, tracking support and component tolerances. Compare paired states and
survival statuses, with integration refinement adequate to resolve the defect.
Exact feature models and reflection averaging require a supported finite
assessment. This remains a measured approximation assumption, not a theorem
about an arbitrary field. A field-parity plot alone is insufficient.

If the measured asymmetry matters, do not discard the opportunity silently or
force exact equivariance. Compare an unconstrained MLP with an equivariant main
branch plus `symmetry_breaking: true` and explicit `breaking_hidden_widths`.
This model accepts honest negative map assessments and costs two networks; its
complete output has no exact reflection guarantee. If only a subgroup is
supported, assess that subgroup separately. Offsets, dipole contamination,
reference choice and asymmetric support can change eligibility.

## Bases, limiting cases and canonical structure are separate questions

Compare an eligible native theoretical residual with general/affine controls.
`assess_native_theory_base` binds the approximation and range to the actual
native planes. A paraxial quadrupole is an interpretable approximation to the
dominant focusing; fringes, finite slopes and higher multipoles remain learnable.
A fitted polynomial degree is a hyperparameter, not a field eigenmode. Keep a
general neural control and inspect whether a step-zero fitted base won.

If all fields scale together to zero, the native map must approach its drift
limit. After verifying that the theoretical base supplies that limit, optional
`residual_gate: coupling` makes its neural correction vanish at zero coupling.
The coupling is divided by its training RMS. This is inappropriate for a fixed
background field or additional independent controls. It is refused on a fitted
base, whose zero-strength map need not be correct. Compare gated and ungated
models under matched conditions when attributing a gain to this constraint.

Slopes `dx/ds,dy/ds` are not canonical momenta. Read the coordinate/scaling and
model-structure guides before choosing canonical layers or a symplectic loss.
Canonical boundary/gauge eligibility is independent of reflection eligibility.
Canonical layers use their contract's common q/p scales. Do not pass the MLP's
`input_normalization` option to that family; it is refused rather than ignored.
Use the installed canonical assessment and refined pullback diagnostic; record
an unsupported backend rather than claiming the target violates Hamiltonian
physics. No full 5D canonical form or unimplemented 6D guarantee is implied.

## Comparable evidence and the example

Match raw-target train-RMS output scales, physical losses/metrics, labels/splits,
seeds, optimizer budgets and complete-forward timing. Use zero-centered input
scaling for the general control of a feature model; otherwise centering is a
confound. Compare base and symmetry separately before assigning causal credit.
Whole-recipe ranking can still select a useful model without identifying causes.
Report empirical tails and accuracy/cost tradeoffs, including invariant features,
base evaluation, projection, breaking branches and public-coordinate conversion.

Run the packaged worked example with
`python -m element_surrogate.quadrupole_example --output ./quadrupole-demo`.
It generates a harmonic-potential field, labels, refinement/map checks, a small
feature MLP and portable export locally. It ships no field, labels or weights.
Its small data size, dimensions, tolerances and training steps are illustrative,
not settings for another magnet. Read its saved report and source for the exact
potential, algorithm and scope. The tested derivation also lives in the proposed
knowledge entry `methods/reflection-features`; do not silently activate it for
a user. Use the existing knowledge-disposition process.
