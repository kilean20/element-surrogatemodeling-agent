# Physical coordinates, numerical scales and loss targets

Establish these three contracts before training. Scaling does not change which
variables are canonical and should not silently change the physical objective.

Use `propose_training_coordinate_scales` on the generated train descriptor or a
numeric NPZ. Supplying `protocol_path` binds the frozen training identity, units
and rows. Read the scope receipt: an unbound legacy NPZ relies on the caller's
native-unit/training designation. Every declared row contributes; the helper
loads one checked shard and batches statistics under `working_memory_bytes`.
Reshard or explicitly increase an insufficient allowance rather than sampling
silently or creating a required full-data NPZ copy. Physical loss tolerances and
canonical eligibility remain separate from this RMS conditioning proposal.

## Slopes are not canonical momenta

An accelerator slope x'=dx/ds is a derivative, not exactly a rotation angle
arctan(x') and not a conjugate momentum. The native plane codec uses direction
ratios in the saved orthonormal basis. At a straight local plane, for actual
delta=(p-p0)/p0 and forward motion:

    pi_x = (1+delta)*xp/sqrt(1+xp^2+yp^2)
    pi_y = (1+delta)*yp/sqrt(1+xp^2+yp^2)

Here pi=p_mechanical_transverse/p0. To normalize to the particle's own p instead,
omit (1+delta). These conventions coincide for fixed p=p0 only. Convert mrad to
rad before applying the nonlinear relation. The inverse has denominator
sqrt((1+delta)^2-pi_x^2-pi_y^2); refuse a nonpositive radicand, never clamp it.

For an untwisted curved reference, write dr/ds=g*t+xp*n+yp*v. Then the denominator
is sqrt(g^2+xp^2+yp^2). The metric g includes the declared signed curvature and
offset. Its sign formula depends on the chosen normal (for t'=kappa*n,
g=1-kappa*x). Do not infer it from a magnet name or equate a derivative along
curved reference s with a straight-plane direction ratio. Torsion/roll needs the
full declared chart, not this simple metric formula.

Canonical P=p_mechanical+q*A depends on signed charge, vector-potential gauge,
reference momentum and coordinate convention. A small-field boundary is not an
exact zero-field/gauge proof. The restricted canonical-shear backend has its own
conservative eligibility test. For nonzero normal B, retain the magnetic
two-form or supply a tested gauge transformation; do not apply a bare J test to
slopes. A 5D map has no nondegenerate canonical J. A 6D test needs the correctly
conjugate arrival/energy pair and its units/sign, not a renamed momentum column.

The package's tested `slopes_to_mechanical`/inverse in `element_surrogate.scaling`
make the metric and actual delta explicit. These functions do not infer A or
provide canonical eligibility. Background: [Tong, Hamiltonian formalism](https://davidtong.org/teaching/classical-dynamics/dynhtml/S4)
and [Lindberg and Borland, dipole fringe maps](https://doi.org/10.1103/PhysRevAccelBeams.26.114001).
The explicit metric and scaling statements here follow by differentiating the
reference-coordinate embedding and pulling back the two-form.

## A principled default numerical proposal

Before labels exist, use the approved proposal distribution as a provisional
magnitude estimate: a uniform symmetric interval [-a,a] has RMS a/sqrt(3).
State that assumption; survivor conditioning can change it. Convert slope
ranges through the actual momentum/metric convention for a mechanical-p scale.
After generation, replace provisional scales with training-only estimates and
freeze them. Do not infer physical error tolerances from those magnitudes.

Use `propose_training_coordinate_scales` on the training cohort from the frozen
protocol. Inputs use training mean/std. New generic neural studies explicitly
use `output_normalization: train_rms`: the raw output RMS, computed before any
base subtraction and shared across all matched base/symmetry ablations. For 6D,
normalize the arrival increment; conserved delta is passed through exactly.
Zero output RMS falls back to matching train-input RMS; a wholly zero component
uses one native unit and is explicitly reported as degenerate. Review that
case. No validation/test target is used to fit a scale.

The proposal also gives common pooled transverse q and mechanical-p/p0 RMS
magnitudes. They are useful starting scales for an eligible canonical backend,
not proof of eligibility. Ask for user-provided characteristic ranges/emittance
or numerical scales when the data are degenerate, strongly anisotropic or have
rare tails. Never guess units, a gauge or physical accuracy from sample spread.
User overrides use `output_normalization: explicit` with positive
`output_scales` in native output units, recorded in the protocol.

For canonical z=D*z_normalized, the normalized two-form is D^T*J*D. Unequal
products q_x_scale*p_x_scale and q_y_scale*p_y_scale require this weighted form.
Using common q0 and p0 across pairs makes it a common scalar times J, so the
ordinary preservation test is equivalent; it does not make D a canonical change
unless that scalar is one. `scaled_canonical_form` tests this pullback explicitly.

## Match the experiment to the question

Physical `loss_scales` remain the intended coordinate error tolerances. Numerical
output scales condition optimization independently. Keep common truth, setting
splits, native coordinates, seeds, batches, optimizer/tuning opportunity, budgets,
physical loss/metrics and full-map timing. A theoretical base and a symmetry
projection need separate matched on/off comparisons. Include a train-affine
control. A strong result for one whole recipe is not an isolated causal effect.

Old v1 configurations with no output_normalization retain legacy loss-scale
parameterization for exact checkpoint compatibility. New generated-study setup
records train_rms explicitly. Changing normalization requires new training or a
tested parameter-conversion path; never reinterpret an old checkpoint silently.
