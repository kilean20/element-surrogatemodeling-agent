# Choose a structure from evidence

The objective is a fast native-plane single-element map. Compare fitting error,
tail errors, setting generalization and the cost of the complete prediction,
including any base, symmetry projection and coordinate conversion. A structural
prior can reduce approximation work, but can also bias a model or cost more than
the accuracy gained. No family is a universal winner.

## General and residual controls

The static-map neural backend reduces scalar field amplitude and effective
rigidity to their ratio. For 5D/6D it also uses momentum deviation, conserves it,
and for 6D predicts an arrival increment independent of absolute arrival.
One static field shape is supported. A non-scaling current-dependent family
requires another backend and another parameter/field-family contract.

Use a general tanh MLP control. Train-only affine/polynomial bases can isolate
dominant low-order response; polynomial degree and feature/memory budgets are
declared hyperparameters. They are approximation bases, not Maxwell eigenmodes.
Train-only normalization and base fitting never inspect validation/test targets.
New generated studies record raw-target train-RMS output normalization, before
base subtraction, independently of loss tolerances. Use matched base/symmetry
on/off ablations and a train-affine control before assigning causal structure
credit. Read [coordinate and scaling rules](coordinates_and_scaling.md) for
canonical momenta versus slopes, defaults, overrides and weighted two-forms.

`assess_native_theory_base` creates a case-bound contract for aligned straight
native planes. `base: theory_drift` is exact for a field-free straight drift;
`base: theory_quad` uses a declared local gradient and the paraxial ideal matrix
over the full native span for 4D/5D. It approximates fringes and nonlinearities;
the neural residual learns their contribution. Six-dimensional drift bases use
actual momentum/mass for time, independent of effective rigidity. The tool
rejects offset/tilted native planes that need another base. Any effective-length
placement adjustment belongs outside this element API.

For the quadrupole base supply the same ranges/settings and optional continuous
coupling interval as the case. Its double-precision portable matrix kernel
uses a degree-12 series in the reduced squared phase and matrix squaring. The
control/momentum bound determines scaling; each reduced argument is at most one,
with next Taylor terms below `1/26!` and `1/27!` before squaring. The bound,
native length and scaling are verified against the assessment. Training inputs
outside that bound are refused. This numerical kernel avoids an exporter-specific
trigonometric precision change and retains finite derivatives at zero strength.

## Symmetry

Field parity suggests a hypothesis, not automatic map equivariance: the fixed
reference, regions, controls and both native planes must also transform properly.
Use `assess_map_symmetry` with the explicit closed reflection group, input box,
settings, seeded probe count and component limits. It tests paired maps/statuses
and saves a source/reference/region-bound assessment. Put the assessment and its
canonical JSON SHA-256 in the candidate; a loose hash is refused. State the finite
probe scope and approximation limits. A generic neural reflection projection
enforces that assumed equivariance, without proving symplecticity.

## Symplecticity and coordinates

For slopes, mechanical momenta normalized to the particle momentum are
`pi_x=xp/sqrt(1+xp²+yp²)`, `pi_y=yp/sqrt(1+xp²+yp²)`. With canonical
`P=p+qA`, the transverse magnetic two-form is
`omega = dx wedge d(pi_x) + dy wedge d(pi_y) - (Bn/Brho) dx wedge dy`.
The signs assume the internal right-handed plane frame and signed `p/q`.
Derivation: pull back `dx wedge d(Px)+dy wedge d(Py)`; the vector-potential
terms combine into `-Bn dx wedge dy`. Thus mechanical momenta give a standard
canonical form only if the normal field is zero on that plane (or a separate
canonical/gauge transformation is supplied).

`analyze_transverse_symplecticity` compares the pullback on three or more finite
difference resolutions, including the reconstructed normal field at the actual
entry/exit positions. It uses user-declared common position/momentum scales.
It reports a finite-cohort diagnostic, with losses/unsupported probes refused.
For 5D/6D it tests fixed-momentum/arrival transverse slices. A 5x5 canonical J
does not exist; a full 6D check needs a correctly conjugate longitudinal pair.
Refine tracking/reconstruction and differentiation separately before attributing
a defect to the learned structure. Accurate RK4 trajectories are not exact
symplectic integration.

The optional `model_family: canonical_shear` candidate uses 4D coordinates and
alternating `q += grad_p T(p,c)` / `p -= grad_q V(q,c)` layers, where each scalar
potential has one tanh hidden layer. Explicit analytic gradients permit ordinary
training and portable export. `hidden_widths: [width]` and `shear_pairs` are
declared study parameters. This is a restricted gradient-module construction
motivated by [SympNets](https://arxiv.org/abs/2001.03750), rather than a claim that
all symplectic networks have the same cost/accuracy. See also
[Chen and Tao](https://proceedings.mlr.press/v139/chen21r.html) for the separate
generating-function approach.

Use `assess_canonical_shear_coordinates` with the case ranges/settings, common
`q_scale_m`/`p_scale` and rationale. Current eligibility is deliberately sufficient
and conservative: both native-plane normal components must be exactly zero at
every supplied node throughout the grid. That proves the condition for the
linear component-wise reconstruction without a boundary approximation. It can
reject a field whose normal component vanishes only on the boundary; this rejects
this backend's proof, not the field or tracker. A nonzero normal component needs
a canonical transformation/gauge construction not provided by this backend.

The model preserves the canonical form up to arithmetic error where its forward
momentum chart remains real. No clipping is allowed. Reflection averaging and
an additive vector residual are refused, because neither generally preserves
symplecticity. Use a generic candidate if the eligibility condition fails.
Do not claim complete 5D/6D canonical constructions from this 4D candidate.

## Study decision

Keep the same native labels, split/group policy, dimensional loss scales,
validation metric, seeds and full-map timing contract across candidates. Choose
configurations through validation, then evaluate one locked model on test.
Discuss whether the measured numerical-label floor makes a more complex model
meaningful. A structural diagnostic and a fitting score answer different questions;
report both. Record rejected structures and their physics/resource reasons so
the next session does not silently repeat them.
