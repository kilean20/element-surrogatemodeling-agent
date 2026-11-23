---
id: strength-error-and-effective-rigidity
title: A magnet's strength error as an effective rigidity
kind: method
status: active
date: 2026-10-31
replaces: null
approved_by: user
keywords: [strength error, field error, rigidity, effective rigidity, scaling, training range, dispersion]
summary: >
  Because a static magnetic trajectory depends only on B/(B rho), a uniform error in a
  magnet's strength is equivalent to tracking at an effective rigidity. When that holds, how
  to convert between the two, and when it does not.
evidence:
  - source: derivation in this entry
    supports: the equivalence, the conversion formula and the asymmetry of the ranges
    verified: true
tools: []
tests: [tests/test_lorentz.py, tests/test_precision.py]
---

## The equivalence

The equation `dt/ds = t x B / (B rho)` (see `lorentz-force-tracking`) depends on the field
and the rigidity only through their ratio. A field scaled uniformly by `(1 + e)` at rigidity
`R0` therefore gives the same spatial trajectories as the nominal field at

    R_eff = R0 / (1 + e),   so   e = R0 / R_eff - 1,

for the same launch conditions and output frame.

## Consequences

- A symmetric range of effective rigidity is not a symmetric range of strength error:
  `R_eff` from `0.8 R0` to `1.2 R0` covers `e` from `+25 %` to `-16.7 %`. State the range in
  the quantity the user cares about.
- Effective rigidity is a local control of one magnet, not the beam energy elsewhere.
- Training data meant to cover a strength error tracks particles of several rigidities
  through one fixed field and one fixed reference; it does not re-fit the reference for each
  rigidity.

## Where it does not hold

Only uniform scaling is covered. It does not describe a change in the shape of the fringe
field, iron saturation, hysteresis or misalignment, and it does not hold for time of flight
or anything that depends on the actual energy (RF, longitudinal dynamics).

## Implementation

The scaling invariance (`B` and `B rho` scaled together leave trajectories unchanged) is an
implemented control in `tests/test_lorentz.py`. `contracts.StaticMagneticCodec`
requires actual momentum and rest mass for time; it never derives speed from the
effective rigidity used to represent field error. Static magnetic momentum deviation
is an exact pass-through coordinate.
