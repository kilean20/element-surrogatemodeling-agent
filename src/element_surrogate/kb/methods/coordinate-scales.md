---
id: coordinate-scales
title: Physical coordinate conventions and numerical scaling
kind: method
status: proposed
date: 2026-11-02
replaces: null
keywords: [canonical momentum, slope, Frenet frame, normalization, loss scale, ablation]
summary: Separate mechanical/canonical variables, train-only numerical scales and physical error targets before any surrogate study.
evidence:
  - source: https://davidtong.org/teaching/classical-dynamics/dynhtml/S4
    supports: canonical electromagnetic momentum includes vector potential; canonical transformations preserve the two-form
    verified: true
  - source: https://doi.org/10.1103/PhysRevAccelBeams.26.114001
    supports: dipole canonical coordinates depend on reference and vector-potential convention
    verified: true
  - source: tests/test_scaling.py
    supports: forward slope/momentum chart, curved metric, weighted scale pullback and training-only normalization controls
    verified: true
tools: [propose_training_coordinate_scales]
tests: [tests/test_scaling.py]
---

An angle-like slope dx/ds is not a canonical momentum. First identify the native
plane/reference chart, actual p0 and delta, signed charge and gauge/boundary
condition. Then choose numerical scales separately from physical error targets.
The installed workflow reference `coordinates_and_scaling.md` gives the exact
conversion, metric limits and weighted two-form. Common train-output RMS is a
default numerical proposal, shared before base subtraction across matched
ablations; it does not turn slopes into canonical variables. Unknown units or
gauge require clarification. No previous magnet's numeric scale is a default.
