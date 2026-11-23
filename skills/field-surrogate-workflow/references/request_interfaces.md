# Installed request and public inference contracts

Run `element-surrogate request-example cards` or `public-4d`, `public-5d`,
`public-6d` to print JSON. Edit case-specific values before use; these are shape
examples, not scientific recommendations or permission records. On Windows use
`Set-Content -Encoding utf8` when redirecting text on older PowerShell versions.

## Regions and longitudinal origins

Transport `support` and `trajectory_region` accept legacy region inputs or an
explicit request. Both normalize into the same `Region`. For a bend, for example:

```json
{
  "shape": "rectangle",
  "radius_m": 0.012,
  "half_height_m": 0.008,
  "coordinate_basis": "nominal_path",
  "bend": {"radius_m": 0.8, "angle_rad": 0.4, "arm_m": 0.1},
  "longitudinal_bounds": {
    "origin": "path_entrance", "min_m": 0.0, "max_m": 0.52
  }
}
```

These synthetic numbers show the shape only. For this example L = 0.52 m, so
the entrance interval [0,L] normalizes to midpoint bounds [-0.26,+0.26] m.
Use `describe_region_request(region)` to check resolved conventions before field
loading. `Region.to_request(origin=...)` serializes accepted inputs and
`NominalBend.to_request()` omits derived description fields.
The preview reports whether nominal entrance/midpoint/exit axis points are
included. A body-only analysis region can intentionally exclude the ends;
do not use that region as a full-path transport restriction by accident.

| Basis | Explicit origin | Legacy z_min_m/z_max_m |
|---|---|---|
| laboratory, without bend | laboratory | Laboratory z in metres |
| nominal_path, with bend | path_entrance or path_midpoint | s_nominal - L_nominal/2 in metres |

Do not mix explicit longitudinal_bounds with legacy z_min_m/z_max_m. Null bounds
are unbounded. `radius_m` is the circle radius, square half-width or rectangle
horizontal half-width; rectangle half_height_m is vertical. `as_dict()` reports
descriptive/derived fields and is not an input serializer. Region validation
occurs before hashing/loading transport field data.

Support constrains accessed interpolation nodes; trajectory_region constrains
particle centres. Neither establishes a mechanical aperture. A nominal path
defines these region offsets; entrance/exit phase space follows the separately
declared native or traced-reference frames. A lab grid x, nominal-path horizontal
offset and traced-reference x generally differ.

## Native inference

The native release directory contains `contract.json`, `model.pt`, `model.onnx`.
`ElementSurrogate` accepts native states followed by effective rigidity in T m
and field scale. Its actual charge/momentum, fixed input/output reference frames
and arrival definition are bound in the physics contract. Effective controls
describe magnetic coupling and do not replace actual momentum in energy/time
conversion. Never refit the reference for each particle or strength-error sample.

`export_public_surrogate_interface` takes the native **contract.json file**, a
request JSON file and an unused output JSON filename. The request declares:

| Field | Meaning |
|---|---|
| `state_units` | One compatible unit per native state coordinate in native order. |
| `fixed_controls` | Scalars fixed for the whole interface; callers cannot override. |
| `default_controls` | Scalars used when a non-fixed control is omitted. |
| `control_bounds` | Explicit finite `[low, high]` for each effective control. |

Both controls are named `effective_rigidity_tm` and `field_scale`. A non-fixed
control without a default is required on each call. Public scalar or per-row
values must satisfy these raw bounds and the underlying native contract.
There is no universal rigidity/strength range. Bind ranges to the approved case.
Units supported are m/mm/um, rad/mrad/urad, s/ns/ps and dimensionless `1`.
Changing time into path, energy into momentum, or slopes into canonical momentum
requires the corresponding coordinate contract; a unit view cannot do that.

```python
from element_surrogate.public_interface import PublicElementSurrogate

model = PublicElementSurrogate("release", "public_interface.json", backend="onnx")
# Shape and units come from the bound interface, not this comment.
result = model.predict(
    states, controls={"field_scale": strengths}, numerical_support_established=True
)
```

Fast inference requires independently established path/stencil support. For
arbitrary proposals, `predict_checked(states, transport, controls=...)` delegates
to the identity-matched native tracker, returns predictions plus stop statuses,
and costs full tracking. Outside raw-control or native bounds has status -1 and
NaN output. This does not identify mechanical apertures. No inverse drift,
effective-length placement, orbit transformation or learned parameter is added.

Card requests have `model.name`, optional `model.frontmatter`, `model.sections`
(normalized pinned template heading to prose), and `datasets.field` / `particle`.
Dataset `frontmatter` uses the separately pinned Genesis schema. Empty draft
examples are intentionally incomplete; they cannot pass full documentation review.
Use the packaged Genesis introspector/guidance and human facts to complete them.
The metadata JSON Schema is a local preflight, not a replacement for LinkML or
registry validation. Preserve exact failures and do not invent a publisher.

Any supplied dataset licence must match that dataset's confirmed terms, even
when its reusable capability is No. A model frontmatter `license`, when supplied,
is the confirmed weight terms' exact `spdx_id` (or exact `name` if no identifier
was supplied). Unknown rights omit licence claims; the narrative records pending
facts. No field/data/software licence supplies the weight terms.


## Model-card input contract extensions

`request-example cards` includes nullable physical_provenance.longitudinal_span
and model.public_interface. A known longitudinal span supplies input_m, output_m,
definition and source_reference. A public interface supplies native_contract_path
and interface_path; the card tool verifies selected-model/protocol/native-byte
binding before converting units. Without it the table correctly documents the
native model's units. The card always records boundary-origin separation,
design/traced lengths, source-grid facts and beam/control units/studied bounds
as available. model_input_contract.json and model_physical_provenance.json retain
exact values; no particle or vector payload is loaded to invent ranges.

model_length_m is the longitudinal native station span; input_output_separation_m is separately computed from the frame origins. The former is null when its coordinate convention/stations are unknown; a chord is not used as a fallback.
