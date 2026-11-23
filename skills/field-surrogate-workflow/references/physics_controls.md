# Physics controls and external current calibration

Use physics parameters with explicit definitions/units as element-map inputs:
field multipole coefficients, integrated strengths or a declared field/rigidity
coupling. For a normal quadrupole, examples are B2 in T/m or k=B2/Brho in m^-2;
state the adopted multipole convention. Do not expose amperes merely because a
source file was calculated at a known current.

The default boundary is:

    current -> external calibration -> physics strength -> native element map

An external caller owns current-to-field calibration and its uncertainty. Missing
calibration does not block a map defined on physics strength. Record known
calculation current in the field/model cards as source provenance, or explicitly
record unknown. Metadata is not an additional required model coordinate.

Several excitation fields can reveal saturation, fringe or multipole-shape
changes. Analyze those changes and decide whether a physics strength alone
identifies the field family. Such data can support B(r; B2), or additional shape/
history controls if necessary; they do not automatically require current as the
public input. A single fixed-shape rescaling does not establish arbitrary-current
or hysteresis behavior. Keep source-shape uncertainty separate from external
calibration uncertainty.

Include current or an internal calibration only when the user requests that scope
and available excitation/calibration data support it. Explain and record the
resulting contract. Do not make future users inherit a facility calibration or
infer current-dependent physics from one solve. Continue the useful physics-input
map with declared limitations when further fields/calibration cannot be supplied.
