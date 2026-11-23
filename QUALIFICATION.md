# Test results and limitations

**0.2.0rc29 is qualified at the recorded setup/adoption scope for a supervised
research preview. Full unattended production is not qualified.**

ESA's runtime, five workflow skills, physics guidance, benchmark procedures and
pinned Genesis resources are supplied locally. Each user's installation checks
its own runtime and source identities. Each scientific case also needs field,
tracking and model validation.

The tools connect to the AI client through the Model Context Protocol (MCP).

## Current executed checks

| Environment and scope | Observed result |
|---|---|
| Windows, Python 3.11.2: focused installed suite | **75 passed**, covering setup, dependencies, knowledge, card and benchmark resources |
| Windows: fresh full CPU source-folder installation | Passed in a new private environment, without a preinstalled ESA/Genesis or uv; doctor, numerical smoke and real local MCP call passed |
| Windows: client files and preservation | Both client configurations checked; 72 managed files verified; original documents/skills and unrelated settings preserved; same-version repeat kept protected files and all 126 dependency versions unchanged |
| Linux, Python 3.13.5: fresh native Codex, new project | Supplied short prompt led to full CPU setup; five skills registered; **99 tools** discovered; live native `inspect_file` succeeded after one ordinary reopen |
| Linux: fresh native Codex, existing project | Separate `cards` runtime installed; original code, data, conventions, skill and frozen scientific environment preserved; live native `inspect_file` succeeded after one ordinary reopen |
| Bundled resources | 185 installed runtime/skill payload members matched source; 27 Genesis and four benchmark-builder pinned assets verified; independent guidance/introspection checks worked without global Genesis |

The two Linux cases used separate profiles and an OS filesystem boundary hiding
the original research repository, global Genesis skills and earlier case answers.
Native client: Codex CLI `0.166.0-alpha.5.1`; requested model/effort:
`gpt-6.1-sol` / `xhigh`. The catalog supported that selection; no separate
provider-side routing attestation was returned.

The existing project was manufactured: a standard-library 1 m drift map and two
checked CSV rows. Its regression and frozen dependency list remained unchanged.
This was an adoption/preservation test, not a surrogate accuracy benchmark.

See the [portable onboarding report](docs/qualification_results.md) for prompts,
timings, recoveries and interpretation. Current publication preparation changes
documentation, issue guidance, Git byte-preservation rules and delivery checksums;
it leaves the tested numerical runtime, setup helper and scientific skills intact.

## What users should expect

Setup creates a private environment, installs the local package, registers project
tools/skills and records checks under `records/setup/`. It does not change global
Python packages or global client settings. Inspect `records/setup/latest.json`
after your installation.

The helper's direct Model Context Protocol (MCP) check is distinct from your AI
client's loaded tool catalog. Normal approval or an owner-terminal command may
be needed for protected client configuration. Reopen when asked and have the AI
confirm a live ESA tool call. No new field/model is qualified by installation or
by a zero-field numerical smoke test.

## Important limits

- The fresh native onboarding conversations were **Linux/Codex only**, one run
  for each entry route plus reopen. These do not establish native Claude Code or
  Windows LLM behavior, or every existing-project conflict. Client configuration
  checks have a narrower scope than a native conversation.
- No new real-field campaign, training population, trained model, accuracy/speed
  result or general autonomous scientific judgment was evaluated in these setup
  tests. Earlier scientific studies retain their exact release, field, domain and
  intervention scopes.
- GPU/HPC setup and throughput are not qualified by these CPU checks. Bounded
  field-processing windows do not impose a process-wide RAM cap; inspect and
  measure the intended workload.
- Supported static-field 4D/5D/6D coordinate contracts have backend eligibility
  limits. Electric/time-varying fields and a general 6D canonical/gauge map need
  additional support. See [scientific/runtime scope](docs/deployment.md).
- Numerical research bounds do not determine a mechanical aperture. Field and
  integration convergence, coordinate/reference conventions, model convergence
  and deployment timing must be checked for the actual case.
- Live Google OAuth/feedback delivery and an automated authorization helper
  remain unqualified. Missing cloud feedback does not stop local work or learning.
- Content/privacy checks reduce risk; they do not prove confidentiality. The
  reviewed software-only copy contained no private case data, model weights,
  credentials or machine-specific client settings. A used workspace needs its
  own review before sharing.

## Earlier evidence

The rc28 predecessor passed **851 Windows tests with two platform-specific
skips** on Python 3.11.2. That result belongs to its generated-workspace layout;
it is not a full rc29 regression run. The separately executed rc29 checks above
must not be relabeled as that older whole suite.

An early rc29 verifier incorrectly expected a source-only skills README inside
the installed payload. It was corrected to use the declared installable skills;
regression and real MCP checks passed. A fresh existing-project native run also
retained and recovered from an ordinary protected-config failure through normal
approval. Neither incident required changing the scientific kernels.

Tests, source identities and failures are retained by the maintainer. The public
summary omits account/session records, local machine paths and raw private logs.
No institutional approval, user-artifact ownership or automatic sharing authority
is inferred from these software checks.
