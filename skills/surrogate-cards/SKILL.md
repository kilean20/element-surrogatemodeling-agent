---
name: surrogate-cards
description: Prepare Genesis documentation for a magnetic element surrogate, its source field and particle-map dataset. Use when creating or reviewing model and data cards with evaluation and governance provenance.
license: Apache-2.0
---

# Document a surrogate release

Use `read_genesis_guidance` to read the packaged pinned datacard-generator and
card-eval-updater instructions, schemas and field/body guides. No host Genesis
installation is required. `prepare_genesis_release_cards` uses these assets and
the optional `element-surrogate[cards]` runtime. Pin and record the schema versions; the
Genesis datacard and registry xCard extension versions can differ. If a required
validator or governance fact is unavailable, produce a clearly labelled draft
and report the exact missing item; do not assert conformance.

Create separate cards for the original field dataset and generated particle-map
dataset. Their contents, producers, processing, access, uncertainty and intended
uses differ. Link particle labels to the exact field/reconstruction/reference/
qualification records, and link the model to that particle dataset. A collection
overview can link both cards but does not replace their descriptions.

For supplied particle data use `prepare_supplied_genesis_release_cards` and the
workflow's [supplied-label guide](../field-surrogate-workflow/references/supplied_labels.md).
It binds the imported cohort/source declaration and the same selected-model
evidence without requiring or inventing a local generation ledger. Record
producer-declared qualification and explicit previous use of each split.
Previously viewed test data support a regression, not a new untouched test.
Read the study's physics review and unresolved topics before describing model
motivation. A recorded deferral must not become an explored structure in prose.

Document these datasets during the workflow, before model fitting finishes.
`prepare_genesis_dataset_draft` accepts a field source or the committed particle
generation.json, plus explicit metadata. It uses the same bundled introspection,
full data-card body and raw validator as the release path. An interrupted or
partial particle dataset is labelled with its actual lifecycle/counts. Store
each draft outside its input dataset and read all findings. Do not wait for
complete generation to identify missing governance/metadata facts. The later
full release tool still requires complete generation and selected-model evidence.

The model card explains the full native forward map, architectural motivation,
parameter reduction, exact conserved coordinates, training and validation-only
selection, benchmark inputs/units/tails, speed workload, numerical support,
loss conditioning and export checks. Extract metrics by script from saved
records; never retype or estimate them. Distinguish compatibility, fitting,
integration and physical-field evidence. Do not claim canonical symplecticity
for slope/time coordinates without the corresponding canonical transformation.
Keep the pinned updater's generated blocks unchanged, including their prose.
Put regression interpretation in ordinary narrative outside those blocks and
check with the raw pinned validator. A locally adapted renderer agreeing with
itself does not establish upstream rerender conformance.

Include design length, source export grid spacing/counts and field extent,
internal solver mesh and input current used for field calculation. Read the
[physical-metadata contract](references/model_card_physical_metadata.md).
Use the actual modeled field's axes and identity. Record a design-length
definition and source separately from traced-reference travel, effective
magnetic length or field-table extent. Export spacing is not the internal FEM
mesh. Record available source calculation current with its signed amperes,
source and independent-coil/basis-solve scope; runtime scale or effective
rigidity cannot supply missing current. Missing facts remain explicitly
"Not recorded" without blocking local modeling or creating a new approval.
The generator adds this table even with supplied Training data prose, saves
model_physical_provenance.json, and records it in traceability.json. Review both
computed axes and supplied source facts. Do not copy another case's numbers.

Introspect the actual datasets using the Genesis procedure before proposing
capability flags. Obtain human metadata the files cannot establish, including
original authorship, owner/contact, funding, classification, reuse terms and
publisher. A maintainer's affiliation does not establish original authorship or
sponsorship. Preserve pending facts and missing-access limitations explicitly.

Run the selected upstream model/xCard/data validators and retain their versions,
commands and receipts. A benchmark result-block draft is not a full Genesis
card. Exporting a local model performs no registry or remote publication; perform
external publication only under the user's applicable release authorization.

The request contains `model.name`, `model.frontmatter`, `model.sections` (normalized
template heading to prose) and `datasets.field` / `datasets.particle`. Each
dataset has `frontmatter` in the bundled Genesis v1.2 schema, optional `body`
notes, and a client-attested `capability_confirmation` with actor, verbatim user
statement and source reference. Introspect before asking for that confirmation;
Use `introspect_card_dataset` before proposing capability flags;
`read_genesis_guidance` provides the bundled introspection and capability guidance.
Do not copy metadata from another user's case. Unknown facts remain omitted and
the actual strict LinkML findings remain in the saved validation report.

Run `element-surrogate request-example cards` for an installed draft example and
`element-surrogate request-schema` for the local metadata contract. Before the
expensive generation-dependent card step, use `validate_card_metadata`. Each of
the field, particle and model artifacts needs its own `rights_confirmation`.
Unknown rights use `status: pending` and cannot support a reusable Yes. Confirmed
rights record subject, terms, downstream permission, actor, user statement and
source reference; this is caller attestation, not authenticated ownership.
The field licence never supplies generated-data or trained-weight rights.
An explicit empty sponsor list records supplied absence; omitted/null is unknown.
Do not convert unknown facts into empty lists just to pass a check.

Model section keys accept the pinned headings with normalized case, whitespace
and trailing periods. Unknown/colliding keys refuse before output. Read the
returned section coverage, including the rights-controlled license section.
Selected native checkpoint facts are appended even with supplied narrative:
distinguish fitted base coefficients, neural parameters, exactly zero correction,
selected optimizer step and finite budget; never infer convergence or neural
improvement from a step-zero selection.

The pinned data schema has an optional numeric software-version slot. This
generator records its exact prerelease in a versioned local software identifier
without inventing a stable version. Supplied metadata are not silently rewritten;
preflight suggests compatible identities. Keep software-schema findings separate
from unknown human facts and retain raw strict validation. See
[documentation reliability](references/reliability.md).

The tool reports local metadata validity, full pinned model-template heading
coverage, numerical evaluation consistency, each raw upstream data-schema
result, and owner/editorial review separately. Heading coverage does not validate
the meaning of prose or the full registry xCard. Read every generated section.

The tool binds the study to the exact complete generation and actual source-field
directory, extracts metrics from saved selected-model evidence, and writes all
GEAR model-template headings. It uses an explicit particle-map regression harness
enum/prose extension, preserving stock numerical rendering and evaluation
validation. Capability-gated data-card narratives derive from the same metadata.
Read the generated cards and upstream validation-extras guide: schema and heading
coverage do not establish scientific/editorial correctness or release authority.
The pinned upstream schema has a documented publisher precondition bug; retain
its raw strict failure while distinguishing a draft's pending publisher from a
true Approved/Published publisher requirement. Do not invent a publisher to pass.

## User authority over generated artifacts

The current user supplies or determines ownership, authorship, contacts, classification, reuse terms and release decisions for their project and generated artifacts. Record those decisions faithfully; ask when needed. Do not assign the base agent's maintainer, organization or Apache-2.0 license to future users' software, fields, particles or models. Pending metadata does not prevent local research or draft documentation. It prevents only unsupported metadata/reuse/publication claims. Do not invent institutional approvals; apply an obligation only when its source actually applies.


Authorship and contact do not establish specific CRediT contribution roles.
Record optional roles, identifiers, funding and sponsorship only when supplied
or established by an identified source. Omit optional unknowns or mark pending
as the schema permits; ask only for facts actually needed. Do not infer who
performed curation, supervision or original writing from a template or name.

The model generator composes one duplicate-checked YAML metric list around the
pinned updater. Read model_metrics_validation.json and the raw pinned validator
log separately. The pinned validator's token parser may warn about benchmark
names containing spaces even when exact YAML entries exist. Preserve those raw
warnings and check actual metric coverage; readable titles are allowed. This
compatibility check does not waive missing metrics or establish registry or
scientific conformance. Read every narrative against supplied facts and results.


## Model input/output length, units and rough valid ranges

The generator always includes a machine-sourced model input-contract table and
model_input_contract.json. It lists every beam/coupling input, actual units and
rough studied ranges from the complete generation plan; optional public-interface
evidence is bound to the selected native model/protocol and converts units while
exposing fixed/default controls. Do not label SI model inputs mm/mrad just because
a public API or another example uses those units. Preserve 4D/5D/6D semantics.
API bounds, discrete studied setting extrema, qualified coupling and trajectory/
stencil support are separate facts; an input box is not an accuracy or aperture
certificate. Effective rigidity/field scale are not calculation current.
Report model input/output frame origins and their computed spatial separation,
defined explicitly as an origin-to-origin distance. For a bend, this is a chord.
Use optional physical_provenance.longitudinal_span (input_m, output_m, definition,
source_reference) for a declared longitudinal station span; keep design and
traced-reference lengths separately named. Unknown station conventions remain
unknown. These ordinary narrative/frontmatter extensions must not alter pinned
Genesis generated evaluation blocks or invent human governance facts.

Record the native input/output longitudinal stations, their coordinate/path definition and source when defining model planes. Use their absolute difference as model_length_m. Keep Euclidean frame-origin separation in the separately named input_output_separation_m. Do not infer longitudinal span from a bend chord, grid extent, traced travel or magnetic effective length. Gather known technical facts from the case before asking the user; missing station conventions can remain explicitly unknown in a draft, but do not claim full technical documentation ready.
