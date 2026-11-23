# Design geometry and field-calculation provenance in the model card

Record these facts during field intake, then carry them into the model card:

| Fact | Evidence and meaning |
|---|---|
| Design length in m | Supplied nominal element/path definition and its source. State the convention; keep traced-reference path length, magnetic effective length and mechanical body length separate. |
| Export spacing x/y/z and node counts | Compute from the actual source-grid axes in SI. Preserve minimum/maximum steps for nonuniform grids. Summary values may be rounded; exact ranges are saved. |
| Field extent | Source-grid axis minima/maxima with coordinate frame and units. This is not the qualified trajectory region or physical aperture. |
| Internal solver mesh | Solver/project notes or source metadata describing FEM/cell sizes, regional/adaptive settings and units when available. Exported table spacing cannot establish this. |
| Input current used for field calculation | Source header, calculation/project notes, identified calibration/source record or user statement. Record signed amperes and coil/calculation scope. Independently calculated basis coils at unit excitation do not establish a simultaneous powered-coil configuration. |

Inspect available metadata before asking for missing facts. A named human/current
fact requires a source reference; missing facts may stay unknown. Do not infer
current from runtime field amplitude, rigidity, field magnitude, filenames alone
or another magnet. No source-current or internal-mesh fact is required to create
research labels or an honest model-card draft. Their absence is documented.

The installed `request-example cards` includes an optional
`model.physical_provenance` object with null values. For KNOWN facts, this is an
illustrative manufactured request fragment, not a default or real source record:

```json
{
  "model": {
    "physical_provenance": {
      "design_length": {
        "value_m": 1.25,
        "definition": "Nominal path between this example's native boundary planes",
        "source_reference": "Replace with the actual layout/user source"
      },
      "solver_mesh": {
        "description": "Replace with actual solver mesh details and units",
        "source_reference": "Replace with the actual solver/project record"
      },
      "field_calculation_currents": [
        {
          "current_a": 7.25,
          "description": "Manufactured example: independently calculated horizontal coil",
          "source_reference": "Replace with the actual source-current record"
        },
        {
          "current_a": -2.75,
          "description": "Manufactured example: independently calculated vertical coil",
          "source_reference": "Replace with the actual source-current record"
        }
      ]
    }
  }
}
```

Use actual case facts, never the illustrative values or placeholder references.
Signed and zero currents are valid facts. Omitted/null/empty current entries
mean no current record was supplied, not zero excitation or a permanent magnet.
A legacy `model.frontmatter.design_length_m` is retained with missing definition
and source stated; supply the referenced definition when available. Conflicting
frontmatter/structured lengths and nonfinite facts fail metadata preflight.

The generator appends a Design geometry and source-field calculation subsection
under the existing Genesis Training data heading. It cannot be suppressed by
supplied prose for that heading. It computes only one-dimensional source axes,
preserves a provided design length as numeric `design_length_m` frontmatter, and
writes `model_physical_provenance.json` plus the same structured facts in
`traceability.json`. Source identities link the model back to the exact field
used for labels. Model/card and field-dataset narratives should agree.

These are local scientific metadata extensions around the pinned Genesis
template. No upstream schema/template is modified; template/evaluation checks
and full registry/scientific/editorial conformance remain separate.


## Boundary distance and longitudinal stations

The model card records model_length_m as the absolute difference between native
input/output longitudinal stations, with their coordinate definition and source.
For a bend this must follow the declared longitudinal path convention, not its
straight-line chord. The separate input_output_separation_m records Euclidean
distance between the two stored frame origins. Neither substitutes for design,
traced-reference arc or field-integral effective length. Supply
`longitudinal_span: {input_m: ..., output_m: ..., definition: ..., source_reference: ...}`
inside physical_provenance. Its absolute station difference is recorded separately.
Unknown stations do not block an honest draft or erase available boundary geometry.
The generated input-contract table/frontmatter gives actual native or bound-public
beam/control units, per-input studied bounds and joint/discrete support limits.
Optional model.public_interface has native_contract_path and interface_path,
both naming existing immutable export metadata for this selected model.
