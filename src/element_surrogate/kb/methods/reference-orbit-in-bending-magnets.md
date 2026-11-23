---
id: reference-orbit-in-bending-magnets
title: The reference orbit through a dipole or bending section
kind: method
status: active
date: 2026-10-31
replaces: null
approved_by: user
keywords: [reference orbit, design orbit, dipole, bending magnet, nominal path, hard edge, fringe field, Frenet-Serret, Courant-Snyder, closure, symmetric exit]
summary: >
  Three objects that are easy to confuse in a bending magnet (the field file's frame, the
  nominal hard-edge path, and the actual reference orbit), how to identify the reference
  orbit by tracing or fitting, how to report particles relative to it, and pitfalls that can
  move results by more than a surrogate's accuracy target.
evidence:
  - source: TRACK v3.9 manual, element mag3d, https://www.phy.anl.gov/atlas/TRACK/Trackv39/Manuals/Mag3D.txt
    supports: the field box placed by the nominal central trajectory ("not a real trajectory"), and a fit of field strength and reference length for an exit symmetric to the entrance
    verified: true
  - source: "W. Hillert, Transverse Linear Beam Dynamics, CERN Accelerator School, arXiv:2107.02614"
    supports: upright accelerator frame versus strict principal-normal Frenet-Serret convention (Section 4.1)
    verified: true
  - source: derivation in this entry
    supports: the closure test, the design-particle condition and the reporting projection
    verified: true
  - source: tests/test_bend.py
    supports: the nominal path's geometry, frames and path coordinates
    verified: true
tools: [fit_reference_orbit]
tests: [tests/test_bend.py, tests/test_lorentz.py, tests/test_precision.py]
---

## Three objects, kept apart

1. **The field file's frame.** The grid the field is tabulated on. For a bend it is often a
   straight Cartesian box whose axis lies along the chord, with its origin at the midpoint
   of the nominal path; decide it from the field values, not from a label.
2. **The nominal path.** The hard-edge design geometry: straight arm, circular arc of the
   design radius and angle, straight arm. It places the box, the entry and exit planes and
   the regions that follow the beam. It is not a trajectory.
3. **The reference orbit.** The trajectory of the design particle through the real body
   and fringe fields. Inside the magnet it departs from the nominal path, by an amount that
   can be large compared with a surrogate's accuracy target; measure it, never assume it
   is zero.

## Identifying the reference orbit

- **Trace:** integrate the design particle at a fixed field strength. Its exit may carry a
  small offset or angle relative to the nominal exit; keep it.
- **Fit:** adjust the field strength, and in a simulator that integrates in z also the
  reference length, until the exit is symmetric to the entrance. TRACK's `mag3d` element
  does this. Check closure in every coordinate (both positions and both slopes), not only
  the one that was fitted: for a symmetric magnet one parameter closes all of them; for an
  asymmetric one it does not, and a multi-parameter match is needed.
- Save the orbit with what produced it: field identity, strength, design rigidity,
  geometry, reconstruction method, and positions and directions along the path. Refuse to
  reuse it with a different field, strength or reconstruction.

## The design particle

Zero transverse input is the design particle only if its momentum equals the design
momentum the simulator computes, with the simulator's own mass and energy constants. A
relative momentum error moves the exit by about the magnet's dispersion times that error;
in a bend, small conversion-constant differences can therefore shift the exit by an amount
that is easily mistaken for a convention difference. Compute the design momentum the
simulator's way.

## Reporting relative to the reference

- Use the transverse plane of the reference at a chosen station: tangent `t`, horizontal
  axis `up x t` (normalised), vertical axis `t x horizontal`.
- Advance each particle to that plane before projecting; equal path length is not the same
  plane.
- Report positions and slopes in that frame. Say whether a result is in the reference's
  frame or in the nominal (geometric) exit frame: the two differ by the reference's own
  closure residual.

## Configuration change versus off-momentum tracking

Changing the design rigidity changes the reference configuration: a fitting simulator
re-tunes the field for it. Tracking particles of other rigidities through one fixed
configuration is off-momentum tracking. Training data for a magnet with an uncertain
strength uses the second (see `strength-error-and-effective-rigidity`).

## Implementation

The nominal path is `element_surrogate.bend.NominalBend` (tested by
`tests/test_bend.py`). Tracing and fitting are implemented in `reference.py`; `ReferenceOrbit.save/load`
records numeric orbit samples, reporting geometry, reconstruction/field identity,
rigidity, scale and integration settings. Loading requires expected field/scale/rigidity
and rejects mismatches. Static reference planes stay fixed across particle momenta.
No implicit inverse-drift/effective-length wrapping is part of this element contract.


## Frame conventions and limits

The implemented frame is upright: ex=normalize(up cross T), ey=T cross ex.
It is not generally the strict Frenet-Serret principal-normal frame, which is undefined
where curvature vanishes and can use the opposite horizontal sign. Upright frames work
through straight sections and planar inflections, but a tangent parallel to laboratory
up is rejected explicitly. Arbitrary 3D roll/transport is not implemented. All internal
PathFrame bases are validated orthonormal and right handed.

Output coordinates in a simulator may refer to its nominal design geometry rather than
an exactly recentered traced ray. The public TRACK mag3d manual specifies the field box
and reference fitting, not every compiled output routine. Establish output frame
semantics for the actual adapter/build; do not assert universal recentering.
Under an improper orthogonal basis change O, position/direction transform as O*v,
but magnetic field transforms as det(O)*O*B. Charge sign and beam reversal are separate;
do not flip field and charge twice. `contracts.transform_vectors` tests axial behavior.
