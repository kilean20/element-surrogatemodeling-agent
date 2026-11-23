---
name: field-map-readiness
description: Assess units, storage order, Maxwell consistency, symmetries, field support and reference conventions. Use when checking a new magnetic field map before tracking or surrogate generation.
license: Apache-2.0
---

# Assess a field map

Inspect format and size before choosing ingestion. For a large CST HDF5 export,
`prepare_mapped_field_store` streams the complete field into an exact float64
store; it preserves all nodes and units and does not improve solver resolution.
Other readers retain their ingestion costs. Use `read_field_file` and
`check_storage_order` on a representation that fits resources. Establish Cartesian versus
curved-file coordinates from evidence; a curved design orbit does not make the
field grid curvilinear. Confirm units and signed field/charge conventions.

Use `check_maxwell`, `scan_node_defects`, `check_harmonic_nodes`,
`analyze_harmonics`, `check_mirror_symmetry`, `check_field_ends` and
`compare_with_description` where applicable. Explain measured scales, derivative
refinement and noise-floor limitations. Compare exports of the same solved field
separately from independent solver-mesh convergence.

In a current-free nonmagnetic region, divergence and curl vanish up to numerical
error, right up to a material boundary. Across magnetic material, tangential B
can jump; do not apply a vacuum-cell curl test across that interface. A departure
from an ideal quadrupole is not a Maxwell error. Maxwell residuals alone cannot
identify the mechanical aperture or calibrate absolute field uncertainty.

Distinguish four objects: source grid, reconstruction stencil, nominal design
path and traced reference orbit. A particle centre inside a region can still
read outside nodes. Qualify the actual reconstruction and integration at the
intended field/control ranges. A black-box perturbation is corroboration; it
needs a responding sentinel and stated finite-probe scope.

Show validity against candidate region sizes, explain the recommendation and
let the user choose. Ask whether longitudinal variation is desired. Keep supplied
mechanical constraints separate from numerical trajectory/support restrictions;
unknown geometry stays unknown. No previous magnet's clearance or step is a rule.
Before generation, use `collect_field_readiness` and `assess_field_readiness`
to bind diagnostics and their explicit interpretation to the source/support.
Unresolved findings block that workflow; the assessment creates no approval or
physical-validity certificate. Load the workflow's readiness contract when needed.

Large readiness diagnostics use bounded blocks with complete stencil halos and
exact global disk-backed statistics. Declare working-memory and disk budgets;
`collect_field_readiness` accepts `working_memory_bytes` and `scratch_directory`.
The default scratch directory is the evidence output's parent. Avoid a RAM-backed
temporary filesystem for large jobs. The working-array budget is not an RSS or
virtual-address cap: mapped input/scratch pages remain OS-managed. Measure peak
process memory, scratch and runtime separately. A computation/resource error
requires rerunning successfully; it cannot be accepted as a field limitation.
Keep the source resolution and numerical criteria unchanged to solve capacity.

Use `describe_region_request` before expensive transport work when coordinates
are uncertain. Explicit requests state coordinate_basis and longitudinal_bounds
with an origin. Legacy bend z_min_m/z_max_m are midpoint-relative nominal path
length; laboratory z and entrance-based path length are different coordinates.
`Region.to_request` and `NominalBend.to_request` return accepted inputs;
`as_dict` is descriptive. Nominal-path offsets differ from traced-reference
phase space. Read the installed workflow's request guide for the exact JSON.

Exact zero nodes and identical planes are contextual observations: a symmetry
zero or longitudinally invariant field can produce them legitimately. Explain
them using geometry, continuity, Maxwell/node-consistency evidence and exporter
metadata. An `explained_observation` disposition applies only to an observation;
it cannot dismiss non-finite values or derivative inconsistencies.
