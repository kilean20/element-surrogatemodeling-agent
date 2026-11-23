---
id: reflection-features
title: Measured reflection symmetry and single-pass invariant features
kind: method
status: proposed
date: 2026-11-11
replaces: null
keywords: [quadrupole, parity, equivariant, invariant features, symmetry breaking, residual, supplied labels]
summary: Measure map effects before enforcing symmetry, then compare character-gated and unconstrained native maps with honest scaling and cost.
evidence:
  - source: derivation in this entry
    supports: quadratic invariants and character-matched coordinate gates give transverse pair-reflection equivariance
    verified: true
  - source: tests/test_equivariant_features.py
    supports: measured eligibility, 4D/5D/6D algebra, one-call inference, fitting, save/load and portable export
    verified: true
  - source: tests/test_physics_review.py
    supports: field parity alone cannot reject map symmetry, and omitted physics comparisons remain explicit
    verified: true
tools: [assess_map_symmetry, review_model_physics, prepare_supplied_neural_surrogate_study]
tests: [tests/test_equivariant_features.py, tests/test_physics_review.py]
---

Scale native transverse coordinates about the declared reflection center before
forming products. For independent horizontal and vertical pair reflections,
invariants are u1²,u1*u2,u2²,u3²,u3*u4,u4² and unchanged controls/momentum. A neural
function of these invariants is unchanged by the group. Multiply its outputs by
the corresponding transverse pair to obtain the required output signs. Both
position and slope gates are needed. General subgroups use their actual sign
characters; central inversion is not identical to two independent reflections.
An arbitrary mean shift of a reflected coordinate changes the physical center.

Field parity suggests a hypothesis. Assess the complete native map with paired
particles, controls and status checks, then compare its deviations with declared
component error allowances and numerical refinement. A small nonzero field
defect neither proves exact symmetry nor proves the prior unusable. Retain real
asymmetry. The feature model's optional unconstrained breaking branch is a
separate learned correction; its complete map has no reflection guarantee, and
its magnitude is not automatically small. Canonical/symplectic eligibility is
independent of reflection eligibility.

The runnable module `element_surrogate.quadrupole_example` generates a harmonic
scalar-potential quadrupole field and trained/exported equivariant MLP locally.
No example field, training table or weights ship. Its settings are instructional
parameters, not production defaults. The installed workflow reference explains
the measured decision branches, comparable controls and finite study scope.
