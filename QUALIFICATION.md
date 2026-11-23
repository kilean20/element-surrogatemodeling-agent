# Qualification: accumulated evidence and operating scope

**ESA 0.2.0rc29 is a supervised research preview supported by earlier completed
scientific campaigns, numerical regression tests, Windows/Linux software checks
and independent Codex onboarding.** A version change does not erase evidence
for unchanged components.

## Demonstrated scientific workflow

An earlier independent, expert-assisted ESA **0.2.0rc22** campaign completed a
real-field 4D (x, x′, y, y′) quadrupole study using the released tools, making
its own model-study choices. It:

- analyzed field readiness, numerical support and integration-step convergence;
- generated **5,000,000 genuine retained particles**;
- completed **54 model fits**, comparable accuracy/speed studies and selection;
- evaluated the locked model on **400,000 held-out particles**;
- met the declared root mean square error (RMSE) limit for each component:
  **10 µm for positions and 10 µrad for slopes**, with tails reported separately;
- met its measured inference-speed goal; the selected public CPU call measured
  about **1.23 µs/particle**, batch 1024, one thread on an AMD EPYC 7313 host.

The campaign also exercised GPU training on its supplied Linux resource.

The recorded (x, y) RMSEs were 1.525/1.233 µm and (x′, y′) RMSEs were
2.080/2.394 µrad. This is evidence that the workflow can produce an accurate,
fast research surrogate at realistic data scale. Those figures describe that
field, sampling distribution, model and timing setup. Boundary-enriched probes
showed larger errors; the RMSE target was not a uniform maximum-error bound.

An earlier numerical consistency study also compared **all 19,968 saved training
labels** from an independent real-field case against TRACK, with separately
refined integration steps. Every row agreed within the declared 10 µm/10 µrad
criterion. The five-million campaign used the released ESA Lorentz tracker and
its own field-specific refinement; it did not freshly rerun TRACK on five million
rows.

These results remain scientific evidence for the implementation they exercised.
They are not replaced by the more recent, narrower installation tests.

## How that evidence carries forward

A SHA-256 comparison of the exact rc22 campaign wheel against the published rc29
source found **81 of its 90 Python modules byte-identical**. The unchanged modules
include field reconstruction, Lorentz tracking, reference/domain transport, data
generation, neural/model training, theoretical bases, symmetry/symplectic helpers
and benchmark protocol/scoring/study execution.

The changed or added modules concern version/interface handling, readiness
advisories, card-process management, feedback/privacy, project state, local
learning and updates. For example, the readiness entry-point change adds
structured warnings; it does not alter the earlier numerical diagnostic values.
Those changes have their own software and integration checks.

Consequently, the earlier campaign continues to support the current scientific
engine. It was not repeated solely for the later setup/documentation changes.
Retesting should follow the effect of a change: regression and integration checks
for changed mechanisms; renewed numerical or model studies when physics,
reconstruction, tracking, fitting or selection behavior changes. Repeating a
large campaign is not a prerequisite for every documentation or packaging edit.

## Software and platform evidence

| Evidence | Executed result |
|---|---|
| Current published numerical baseline, commit bee4019 | [GitHub CI passed](https://github.com/kilean20/element-surrogatemodeling-agent/actions/runs/45217268499): Windows/Linux/macOS test jobs on Python 3.11 and 3.14, full-runtime and source-folder setup jobs on Windows/Linux, lint, lock and conformance checks |
| This maintainer's Windows machine, Python 3.11.2 | Fresh full CPU source-folder installation passed, plus **75 focused installed checks**, numerical smoke and an actual local Model Context Protocol (MCP) call |
| Windows project configuration/preservation | Both client configurations checked; original files/settings retained; same-version repeat preserved protected files and all 126 dependency versions |
| Fresh Linux/Codex onboarding | Both short user prompts (new/existing projects) passed, including actual native MCP calls after an ordinary reopen; the existing project's frozen scientific environment was preserved |
| Local learning/reopen | An earlier rc27 native manufactured case preserved its accepted coordinate interface and validated local methods across a new client process |
| Installed assets | 185 runtime/skill members matched source; 27 Genesis and four benchmark-builder pins verified; guidance and dataset introspection worked without global Genesis skills |
| Earlier broad regressions | rc28: 851 Windows tests passed, two platform skips; rc27: 827 Windows passes/two Linux-only skips and 829 Linux passes. These are separate recorded runs, not counts to add together or relabel as a fresh rc29 suite |

The native onboarding conversations used Linux/Codex because the test's existing
filesystem boundary could hide the originating repository and global skills.
Windows was tested at installation/runtime/tool level, both locally and in CI.
A fresh Windows Codex conversation and a native Claude Code conversation have
not yet been mirrored; that is a client-interaction coverage gap, not evidence
that Windows or Claude Code is incompatible. See the
[onboarding report](docs/qualification_results.md) for the exact procedure.

The first native source-folder tests used Codex CLI 0.166.0-alpha.5.1, requesting
GPT-6.1 Sol / extra-high effort. Each client must load its connection and confirm
an actual ESA tool call after setup/reopen. The helper's direct protocol check
and a live client call are complementary checks.

## Material operating limits

- **Source truth and numerical validity:** a new field needs its own units,
  reference/coordinate, reconstruction-support and integration-convergence
  checks. A numerically consistent export is not proof of the original solver's
  physical accuracy. Numerical training bounds do not determine a mechanical
  aperture; unknown geometry can remain explicit within a research study.
- **Case-specific model performance:** accuracy and speed depend on the field,
  sampled domain, training, model and hardware. The successful 4D campaign is not
  a universal error bound or a guarantee for a new magnet. Supported 5D/6D
  contracts have analytical/manufactured checks; the cited large real-field
  campaign was 4D.
- **Implemented physics:** the supplied backends address static magnetic fields.
  Electric/time-varying fields and a general full-6D canonical/gauge map need
  additional support. Backend eligibility and reference-frame limits are in
  [supported scope](docs/deployment.md).
- **Resources and supervision:** bounded field windows are not a process-wide RAM
  cap. Measure the intended workload. Expert review of consequential physical
  assumptions and the resulting surrogate remains part of research use. The
  successful campaign used an expert proxy for requirements/resource questions;
  its long generation and fitting jobs ran under the authorized compute budget.
- **Optional cloud feedback:** live Google authorization/upload and an automated
  onboarding helper remain unqualified. Local modeling and learning work without
  that cloud connection.

## Records, changes and user artifacts

The latest source-folder setup tests add evidence about distribution, adoption
and client connection. They do not negate the earlier scientific campaigns or
imply the current version cannot perform them. Current qualification/documentation
corrections leave the numerical runtime, setup helper and scientific skills
unchanged.

Source snapshots, numerical controls, actual conversations, successes and failed
attempts are retained with their dates/releases. Earlier pilots that found memory,
executor or workflow defects remain development/regression evidence. Exact
versions preserve traceability; they are not an automatic expiration date for
unchanged-component results.

The public summary omits field payloads, training rows, model weights, local
machine paths and private account logs. The software-only copy was reviewed;
a used scientific workspace needs its own sharing review. User artifacts keep
the user's ownership and reuse terms. See [shipping guidance](SHIPPING.md).
