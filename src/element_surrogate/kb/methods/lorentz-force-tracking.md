---
id: lorentz-force-tracking
title: Lorentz-force tracking through a static magnetic field map
kind: method
status: active
date: 2026-10-31
replaces: null
approved_by: user
keywords: [Lorentz force, particle tracking, rigidity, Runge-Kutta, path length, field map, interpolation, exit plane]
summary: >
  How to integrate a charged particle through a static magnetic field map: the equation of
  motion in path length with signed rigidity, what it conserves, how to reach an exit plane
  and report failures, and what two trackers must share before their results can be compared.
evidence:
  - source: derivation in this entry
    supports: the equation of motion in path length and the conservation of momentum magnitude
    verified: true
  - source: TRACK v3.9 manual, element mag3d, https://www.phy.anl.gov/atlas/TRACK/Trackv39/Manuals/Mag3D.txt
    supports: a simulator may integrate in z rather than path length and fit its reference first
    verified: true
  - source: "V. N. Aseev et al., TRACK: The New Beam Dynamics Code, PAC 2005 (as cited by a facility study; not read here)"
    supports: TRACK reads 3D field tables by quadratic interpolation
    verified: false
tools: [fit_reference_orbit, check_tracking_convergence, check_adaptive_tracking_convergence]
tests: [tests/test_lorentz.py, tests/test_precision.py]
---

## Equation of motion

In a static magnetic field the Lorentz force does no work, so the momentum magnitude
`p = |p|` is constant. With the unit direction `t = p / p`, the path length `s`
(`ds/dt = |v|`) and the signed rigidity `B rho = p / q`, the equation
`dp/dt = q v x B` becomes

    dr/ds = t
    dt/ds = t x B(r) / (B rho)

Relativity enters only through `p` in `B rho`; adding a separate Lorentz factor to this
path-length form would be wrong. The sign of `B rho` carries the sign of the charge.
Trajectories depend on the field only through `B / (B rho)` (see
`strength-error-and-effective-rigidity`).

## Integration

- Use field-dependent adaptive local error control, with curvature, variation and
  reconstruction-switch limits. No universal step follows from another magnet. The
  RK4 step-doubling implementation is not symplectic: monitor direction-norm drift.
- Convergence is measured, not assumed: repeat with the step halved and report the change
  in every output coordinate, including its maximum, not only an average.
- To end on an exit plane, shorten the last steps so the particle reaches the plane within
  a stated tolerance; do not extrapolate a long step across it.
- Report failures explicitly and keep their output empty: leaving the field table's
  support, a non-finite field or state, exceeding the path budget, crossing the exit plane
  backwards. A particle outside the table is not in vacuum; the table simply has no data.

## Reading the field between nodes

The field map is only defined at its nodes; the tracker integrates a field *reconstructed*
between them (trilinear, quadratic, spline). Different reconstructions are different
fields: step refinement of each converges to a different limit. Before comparing two
trackers, make them read the mesh the same way, or measure the difference that the
reconstruction alone makes.

## Coordinates in and out

States are positions and slopes `(x, x', y, y')` in a transverse frame, with slopes
`xprime = t_x / t_s` taken from the direction, not paraxial angles; they are not
canonical momenta. Building the launch direction from slopes means normalising
`(x', y', 1)`. Which frame the states are measured in is a separate choice (see
`reference-orbit-in-bending-magnets`).

## What two trackers must share before agreement means anything

The same field values and reconstruction, units, field sign and scaling, design momentum
(including the mass and energy constants used to compute it), launch and exit planes,
reference definition and output frame. Agreement at one step size is not convergence:
show both trackers converged before calling them equivalent.

## Implementation

`lorentz.track` is the explicit fixed-step diagnostic; `adaptive.track_adaptive`
is the error-controlled CPU reference. Both use the same path-length Lorentz equation.
`readings.reconstruct` selects total-degree-two by default, with a named 19-node
uniform-grid stencil; other methods require an explicit selection. Unsupported grids
raise an error instead of a silent fallback. Degree two does not impose Maxwell
constraints. `step_convergence` omits order for unequal ratios; adaptive convergence
reports component/path changes and all statuses. Neither is an absolute error certificate.
Analytic controls: exact helices in uniform fields of any direction and either charge
sign, straight drift, and invariance when `B` and `B rho` are scaled together.


## Coordinate and timing qualifications

For a planar outward-horizontal convention with Tprime=-h*ex,
`dx/ds_ref=(1+h*x)*xprime`. A local plane ratio is not generally a derivative
with respect to reference arc length. Mechanical slopes are not canonical momenta;
canonical tests at nonzero boundary field require the vector-potential convention.
The 5D momentum deviation passes through exactly for a static magnetic field.
`contracts.StaticMagneticCodec` advances time/ct/path in 6D from tracked path length.
Time requires actual momentum and rest mass, separately from effective rigidity.
The code has no electric, radiation, space-charge or material model.

A 5D transverse-plus-momentum map is a family of 4D maps with a conserved
momentum-deviation coordinate. There is no nondegenerate canonical symplectic
form in odd dimension: do not apply a purported standard 5-by-5 symplectic J.
Use fixed-momentum canonical 4D tests or a correctly defined canonical 6D lift.
The slope/time codecs by themselves do not supply that canonical transformation.

Physics qualification uses analytic/manufactured controls independently of compatibility
with TRACK. Matching reconstruction and conventions is necessary to interpret differences;
agreement with one tracker is not the definition of correct physics. Nearest-stencil
reconstructions can jump: a face cap/local estimator is not proof of fourth-order
convergence. Repeat independent refinement for each field and operating range.
