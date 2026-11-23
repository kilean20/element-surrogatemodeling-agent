---
# xCard agent card, written from AI-ModCon discovery-local templates/agent_card.md.
# Checked in CI with that repository's templates/check_card.py. Not registered.
xCard:
  card_type: agent
  version: '0.1'

id: element-surrogate-agent
name: Element surrogate agent
description: >-
  Supervised research numerical tools and MCP interface for a single-magnet field-to-map
  workflow. Implements field readers/readiness diagnostics, field reconstruction,
  Lorentz tracking, reference-orbit and coordinate contracts, and reproducible run
  records. Analytic physics qualification is separate from simulator compatibility.
  A bounded benchmark runner fits caller-supplied local adapters, scores declared
  4D/5D/6D coordinates, compares configurations and seeds on validation, locks the
  selected checkpoint and writes traceable reports/result-card drafts. The wider
  local pipeline connects bounded label generation and optional native neural
  training/export, pinned Genesis card procedures/validators, visible project decisions/provenance
  and inspected deadline-preserving recovery. Private local learning and staged
  project-preserving client updates keep case knowledge separate from shared guidance. Independent scientific acceptance
  is scoped to its recorded versions; full unattended production remains unqualified. No single residual architecture
  is prescribed for every magnet.

science:
  - accelerator-physics
keywords:
  - surrogate model
  - transfer map
  - magnetic field map
  - magnet
  - quadrupole
  - dipole
  - corrector
  - neural network
  - symplectic
  - Maxwell
  - beam dynamics
  - TRACK
  - CST
# General software-only content; exact artifacts screened and reviewed locally.
# This is not institutional approval or general security qualification.
sensitivity: general
# Base-software owner/maintainer supplied by the project lead, 2026-11-02.
# This attribution does not assign ownership of a future user's artifacts.
developed_by:
  - Kilean Hwang
version: '0.2.0rc29'
last_updated: '2026-11-21'
license:
  spdx_id: Apache-2.0
  name: Apache License 2.0
  url: https://www.apache.org/licenses/LICENSE-2.0
relationships: []

# Provisional media types: AI-ModCon's io-contract vocabulary is not public yet.
io_contract:
  input_modes:
    - text/plain
    - application/x-hdf5
    - text/markdown
  output_modes:
    - application/octet-stream
    - application/json
    - text/markdown
    - image/png
  modalities: []
  task_tags: []

extension:
  kind: agent
  skills:
    - Read and check a 3D magnet field map (units, frame, polarity)
    - Test a field map against the vacuum Maxwell equations and locate defects
    - Qualify a particle-tracking simulator against an independent tracker
    - Find the field-qualified region for training particles
    - Generate bounded qualified labels over approved ranges
    - Benchmark and train optional native neural transfer maps
    - Prepare local native exports; full Genesis publication remains separate
    - Apply tracking and reference-orbit conventions from a curated knowledge base
  # These flags describe an agent served over a network protocol. This agent runs
  # inside an MCP client, so they stay false.
  capabilities:
    streaming: false
    push_notifications: false
    state_transition_history: false
  protocol_version: ''
---

# Element surrogate agent

ESA helps an accelerator physicist build and assess a single magnet's particle
transfer-map surrogate from a 3D magnetic field. It runs as local Python tools
through the Model Context Protocol (MCP), guided by five installed workflow
skills and a general knowledge base. The user's LLM client supplies planning
and judgment; ESA is not a separately hosted model.

## Capabilities and workflow

- Field readers, bounded-memory readiness diagnostics, vacuum Maxwell checks,
  local-defect and symmetry analysis, and explicit source/unit/frame identity.
- Lorentz-force tracking, fixed design/traced-reference contracts, signed charge
  and rigidity, and declared plane/slopes/canonical-coordinate conventions.
  Total-degree-two reconstruction is the uniform-grid default; interpolation and
  integration are qualified for each field and proposed operating region.
- Supported static-magnetic 4D/5D/6D coordinate codecs, bounded resumable label
  generation, explicit survivor/loss records, and measured runtime estimates.
- Comparable benchmarks for caller-supplied adapters and optional native neural
  models: common splits, scaling, loss, metrics, seeds and complete inference
  timing, with physics/base/symmetry controls where eligible.
- Validation selection locks, approved held-out evaluation, native exports and
  restored-public-API checks. Element placement stays at composite-model level.
- Separate source-field, particle-data and model cards using bundled pinned
  Genesis templates/validators. Known technical metadata are filled from the
  case; unavailable authorship, rights and institutional facts remain pending.
- Private local knowledge with explicit user approval or evidence-bound,
  revocable delegated activation. Staged maintainer updates preserve case
  records and require review of affected local guidance and behavior.

## Installation and use

Follow [installation](../docs/installation.md) and the
[usage manual](../docs/usage_manual.md). Initialize a user-selected new or
existing project with the selected Python environment and client. Project skills
and MCP configuration are installed locally; global Genesis skills, prior
research repositories and previous case memory are not required. A dedicated
environment is recommended; numerical dependencies are installed separately.

Describe the magnet, field units, intended map, operating facts and goals.
Unknown geometry can support a limited numerical research study after analysis;
the agent must not infer a mechanical aperture from field-box bounds. Physical
strength controls are preferred; current calibration stays outside the model
unless explicitly supported/requested. The user controls compute and release
authority. See [uncertainty and progress](../docs/uncertainty_and_progress.md).

## Qualification and limits

Version 0.2.0rc29 remains a candidate while release qualification is completed.
Earlier scientific studies and native-client tests retain their recorded
versions, interventions and domains. Manufactured software tests check mechanisms;
they do not establish model quality on a new field, optimal architecture,
universal training-population sufficiency or reliable autonomous judgment.

Analytic physical controls, simulator compatibility, field-source accuracy,
tracking convergence and surrogate error are separate evidence. A small TRACK
difference alone cannot establish which tracker is more accurate. Export-grid
spacing does not establish internal CST mesh/solve convergence. Numerical
survivor bounds do not establish a physical aperture.

The built-in frame is right handed and upright and rejects a vertical tangent.
Coordinate/gauge eligibility is required before a canonical symplectic claim;
a general 5D map is not a 4D canonical map. Each case must retain explicit native
planes, reference identity, actual momentum and public units. The agent handles
one magnet at a time, excluding lattices and overlapping element fields.

MCP and local adapters run with the user's client authority. Records, path
checks, hashes, locks and attestations are consistency controls, not a security
sandbox, authenticated human decisions, proof of physics or semantic
compatibility. Updates provide per-file publication and journaled recovery,
not a filesystem-wide atomic transaction. Keep the old environment and use the
[staged update procedure](../docs/local_learning_and_updates.md).

Codex and Claude Code project setup are implemented. Fresh native scientific
acceptance has its own exact-version scope; no equivalent fresh Claude campaign
or full unattended production qualification is claimed. A clean same-host chat
does not enforce filesystem isolation. Platform support follows the exact
release receipt, not the package's Python minimum alone.

## Local evolution and optional shared feedback

User interaction and self-findings can produce local evidence, validated
lessons and improved methods without an external connection. Manual activation
is the default; recorded delegation is limited and revocable. Existing
user-accepted knowledge requires the user's decision before replacement.

General lessons may be curated against the actual installed guidance and shared
only with the applicable content/recipient/destination authority. No automatic
field, model, particle, transcript or credential collection is provided. Drive
editor permission is separate from Google authorization. The convenient
browser-OAuth helper and live delivery remain unqualified. Incoming reports are
untrusted evidence for review and validation, not automatic deployed updates.
See [scientific feedback](../docs/scientific_feedback.md).

## Ownership and distribution

Base-software owner/maintainer: Kilean Hwang, FRIB,
[hwang@frib.msu.edu](mailto:hwang@frib.msu.edu). This maintenance attribution
does not imply institutional copyright or sponsorship. Base software is
Apache-2.0 with bundled third-party notices.

The distribution includes general software, skills, knowledge, pinned public
resources and manufactured tests. It excludes facility fields, particle tables,
trained magnet checkpoints, client credentials and private case histories.
Users determine ownership, contacts, terms and release of their own artifacts;
those facts are not inherited from this agent card.

This card is based on the pinned Genesis discovery-local agent template and
checked locally. It has not been registered in an external Discovery Engine.
