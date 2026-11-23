# Element Surrogate Agent (ESA)

**Physics-aware tools and guidance for building single-magnet particle-map
surrogates with Codex or Claude Code.**

ESA helps an accelerator physicist work from a 3D magnetic field map through
field checks, Lorentz-force tracking, training-data generation, model studies,
accuracy/speed benchmarking and documented model selection. Your AI client
supplies the language model and permissions; ESA supplies local tools, five
workflow skills, physics guidance and bundled Genesis card resources.

**Version 0.2.0rc29 is a supervised research preview.** Fresh Windows installation
checks and independent Linux/Codex onboarding passed. Each new scientific case
still needs its own numerical and model validation. See
[test results and limitations](QUALIFICATION.md).

## Download and start

1. [Download ESA from GitHub](https://github.com/kilean20/element-surrogatemodeling-agent/archive/refs/heads/main.zip)
   and extract the folder. This is also available under **Code → Download ZIP**
   on the [repository page](https://github.com/kilean20/element-surrogatemodeling-agent).
   When tagged releases are available, choose one under **Releases → Source code (zip)**.
   You can also clone the repository; Git is not required for an extracted download.
2. Open that folder as a project in an installed, signed-in **Codex or Claude
   Code** client. Python **3.11 or newer**, with `venv` and `pip`, is required;
   the AI can help locate a suitable installation.
3. Paste this prompt:

~~~text
Read SETUP.md and set up the Element Surrogate Agent (ESA) in this project for
this AI client. Choose an appropriate Python environment, verify that the tools
work, and tell me how to provide my field data and begin modeling.
~~~

Setup installs dependencies in a private Python environment and connects this
project's tools and skills. Allow the normal project permissions; reopen the
client if asked so it can load and test the connection. Numerical dependencies
are downloaded separately and can be much larger than this software folder.
No uv, Docker, hosted ESA server or global Genesis installation is required.

**Already have a modeling project?** Keep the ESA download separate, open your
existing project and use the [existing-project prompt](USER_PROMPTS.md#2-adding-esa-to-an-existing-project).
The AI should review and reuse accepted work, preserve its conventions and
working environment, and add useful capabilities. Adding ESA does not require
starting the scientific study over.

After setup, provide your field-file path and describe the task. For example:

~~~text
Build a 4D transverse particle-map surrogate from the field at "<field file>".
Analyze the field first, recommend suitable ranges and accuracy/speed goals,
and estimate the computing time. Ask me about physical facts you cannot establish.
~~~

See [USER_PROMPTS.md](USER_PROMPTS.md) for the copyable prompts,
[START_HERE.md](START_HERE.md) for folder choice and [SETUP.md](SETUP.md) for the
AI's setup procedure.

## Why ESA exists

Surrogate studies need more than fitting a network. ESA is designed to reduce
repeated manual reminders about the scientific decisions that affect a result:

- **Field readiness before expensive tracking:** check units, coverage,
  vacuum Maxwell consistency, reconstruction and measured symmetry.
- **Explicit coordinates and references:** distinguish design and traced
  orbits, handedness, signed rigidity, slopes and canonical momenta.
- **Physics-informed model studies:** consider eligible symmetry, theoretical
  bases and symplectic structure, with matched comparisons that isolate each
  design choice.
- **Comparable accuracy and speed:** use shared data/splits, scaling, losses,
  metrics and timing scope; report tails and measured tradeoffs.
- **Traceable decisions and artifacts:** preserve source identities, numerical
  settings, selection locks, model cards and separate field/particle data cards.

These procedures support expert review. They do not establish that an unknown
field is physically correct or that one architecture is best for every magnet.

## What is included

The local tools cover static magnetic field reconstruction, Lorentz tracking,
reference/coordinate contracts, resumable generation, model comparison and
supported native 4D/5D/6D maps. Eligibility depends on the physical and coordinate
contract; see [supported scope](docs/deployment.md).

The bundled Lorentz tracker is the default independent numerical tool. TRACK
comparison is optional compatibility evidence. Integration steps are qualified
per field and region. Numerical survivor bounds are distinct from a physical
aperture, and native element maps leave lattice placement to composite modeling.

| Location | Purpose |
|---|---|
| `USER_PROMPTS.md`, `START_HERE.md`, `SETUP.md` | Starting prompts and setup |
| `src/element_surrogate/`, `skills/` | Numerical tools, AI-client connection, scientific guidance and pinned resources |
| `docs/`, `cards/`, `tests/` | Manuals, base-agent card and software checks |
| `inputs/`, `data/`, `studies/` | Your inputs, generated particles and studies; created during setup/use |
| `models/`, `reports/`, `records/` | Your models, documentation, decisions and resumable records |
| `.venv/` | Default private Python environment created during setup |

Project lessons remain local. Maintainer updates use a reviewed adoption process
that preserves user knowledge and accepted decisions. Optional feedback requires
content review and sharing authority; live Google authorization/upload remains
unqualified. See [local learning](docs/local_learning_and_updates.md) and
[feedback](docs/scientific_feedback.md).

## Evidence, help and ownership

- [Qualification summary](QUALIFICATION.md) and [independent onboarding report](docs/qualification_results.md)
- [Usage manual](docs/usage_manual.md), [installation details](docs/installation.md) and [documentation index](docs/index.md)
- [Known scientific/runtime limits](docs/deployment.md) and [uncertainty handling](docs/uncertainty_and_progress.md)
- [Contributing](CONTRIBUTING.md), [security contact](SECURITY.md) and [release history](CHANGELOG.md)
- [Maintainer publication guide](SHIPPING.md)

For a bug report, use [GitHub Issues](https://github.com/kilean20/element-surrogatemodeling-agent/issues)
with a minimal sanitized example. Keep
credentials, private field/particle data and unreviewed session logs out of public
issues. Use the private contact in SECURITY.md for security concerns.

The shared software includes no facility field maps, particle datasets, trained
magnet checkpoints or client credentials. Users determine ownership and reuse
terms for their own artifacts. The base software is maintained by Kilean Hwang,
FRIB, [hwang@frib.msu.edu](mailto:hwang@frib.msu.edu), under
[Apache-2.0](LICENSE); bundled third-party resources retain their notices.
