# Physics controls; calibration outside the element map

The default element API takes defined physics parameters with units, such as a
quadrupole coefficient B2 in T/m or k=B2/Brho in m^-2. An external calibration
converts current to those parameters. Missing current calibration is not a reason
to stop modeling a map defined on physics strength.

Known calculation current remains field/model-card provenance. Several excitation
fields can support a shape-dependent field family parameterized by physics
strength; they do not automatically make amperes the public input. A single
rescaled source does not establish saturation or hysteresis behavior. Include an
internal calibration only for an explicitly requested, evidence-supported scope.

The bundled workflow reference `references/physics_controls.md` gives the full
decision procedure. It is installed with the project skills and needs no global
skill or developer memory.
