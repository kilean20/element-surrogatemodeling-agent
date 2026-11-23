---
id: track
title: TRACK, as a source of field tables and of training data
kind: simulator_profile
status: active
date: 2026-10-31
replaces: null
approved_by: user
keywords: [TRACK, simulator, field table, sol3d, sol3c, mag3d, header, units, aperture, reference trajectory, interpolation, step]
summary: >
  What the agent must know before reading TRACK field tables or comparing with TRACK's
  tracking: the table layout and header meaning, units, the aperture sign, how the mag3d
  dipole element places its field and fits its reference trajectory, how it reads the mesh,
  and what to check on the build in use.
evidence:
  - source: TRACK v3.9 manual, element mag3d, https://www.phy.anl.gov/atlas/TRACK/Trackv39/Manuals/Mag3D.txt
    supports: header lines (field type 0 E, 1 B, 2 E and B; symmetry flags), axis lines, storage order with z fastest, the field box geometry, z-based integration and the reference fit
    verified: true
  - source: "V. N. Aseev et al., TRACK: The New Beam Dynamics Code, PAC 2005 (as cited by a facility study; not read here)"
    supports: quadratic interpolation of 3D field tables
    verified: false
  - source: TRACK v3.9 manual, element sol3d (aperture parameter)
    supports: a negative aperture value means a circular aperture of that radius, a positive one a rectangular half-aperture
    verified: false
  - source: tests/test_fields.py
    supports: the reader's handling of the header, storage order, interleaved rows and duplicated rows
    verified: true
tools: [read_field_file, check_storage_order]
tests: [tests/test_fields.py]
---

## Field tables

- Header: a field-type line (`0` E, `1` B, `2` E and B), a line of symmetry flags in x, y
  and z, then three lines `low high count` in cm for x, y and z. Data rows follow with x
  varying slowest and z fastest, three components per row in gauss.
- Type `2` tables interleave an electric row before each magnetic row; a type-`1` table
  has one row per node. Multi-slot tables (one field per coil) carry three columns per slot
  and a first line in another form, which the manual seen does not document.
- Nonzero symmetry flags mean TRACK may unfold the stored part of the field; a reader that
  does not unfold must say so.
- The reader `read_field_file` reports all of this; `check_storage_order` tests the
  storage assumption against Maxwell's equations.

## Aperture sign

A negative aperture value means a circular aperture of radius `|a|`; a positive one a
rectangular half-aperture. Using the wrong sign gives a rectangular survival region where a
circle was meant. (Not re-verified from TRACK's own documentation in this entry.)

## The mag3d dipole element

- The field box lies on the chord: its z axis makes half the bend angle with the incoming
  and outgoing beam, and its origin is the midpoint of the central (nominal) trajectory,
  which the manual calls "not a real trajectory".
- TRACK integrates in z. Before tracking, it fits both the field strength and the reference
  trajectory length, from the design beam's energy, mass and charge, so that the reference
  exits symmetric to its entry; that length is then divided into the requested steps.
- Changing the design energy makes TRACK re-fit the field: a new configuration, not an
  off-momentum particle (see `reference-orbit-in-bending-magnets`).

## Checks on the build in use

- **Output frame.** Track the exact design particle (momentum from TRACK's own constants)
  with zero transverse input. A nonzero output can come from reporting-frame, numerical closure or momentum
  mismatch; it alone does not identify the frame. Compare logged lab positions and
  directions projected into each candidate plane/origin/basis. Verify this for the
  actual build and element; the manual does not prove exact traced-ray recentering.
- **Mesh reading.** TRACK is described as reading tables quadratically; the exact stencil
  is not given in the manual. Infer it with a field whose answer is known before comparing
  with a tracker that reads differently.
- **Steps.** Refine the step count and measure the change; do not assume convergence.
  Check that every run produced its output files, whatever its exit code.
