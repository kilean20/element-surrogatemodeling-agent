# Executor context and scientific recovery

These are cooperating local-job controls. They do not authenticate a human,
authorize new compute, or sandbox arbitrary shell code.

## One process observer per campaign

Use the project's persistent MCP service consistently for scientific job control.
On Linux, a client shell sandbox can use a different PID namespace. A PID number
there cannot establish whether a worker in the persistent service is alive.
The library records observer context in the campaign ledger, separately from the
immutable user resource policy, and refuses mismatched access before process
probes or state mutation. Same-context direct Python use is supported.

Linux process identity uses the boot identity and kernel process-start counter;
wall creation time is informational. A missing or inaccessible process in an
unobservable context does not release capacity or prove work has ended.

Run the full dependency doctor with the exact Python configured in the project's
MCP configuration, inside the actual client environment. An administrator-shell
check can see dependencies hidden from the client. Standard Python, operating
system libraries and the user's client account are external prerequisites.

## Resume and a new diagnostic allocation

`inspect_scientific_job_recovery` and `resume_scientific_job` retain the original
job's remaining allocation, inputs, implementation and calendar bound. Unknown
work after loss of both observers is charged conservatively and cannot auto-resume.

For a terminal failed immutable-input diagnostic, use
`inspect_scientific_diagnostic_retry`. It verifies the frozen predecessor,
previously verified input hashes, inactive owners and released job guard, then
returns a request template and current implementation digest. Fill that template
with a positive wall allocation, resources, a fresh project output root and a
reason. Submit it with `retry_scientific_diagnostic`.

The successor consumes the original campaign's remaining allowance and preserves
all predecessor charges, including uncertain full charges. Any implementation
change is explicit. No old result earns passed credit, no calendar bound resets,
and one predecessor cannot fork concurrent successors. Identical submissions are
idempotent. A changed input or stale inspection requires a new review. The tool
does not invent approval: use existing user authorization or obtain a new decision
if the requested work exceeds it.

Generation, model fitting and held-out evaluation have mutable state and retain
their checkpoint/test-specific protocols. They cannot use generic diagnostic
replay to discard partial work or reopen a frozen test.

## Legacy adoption and handoff

Do not edit a legacy ledger to create provenance. `bind_scientific_executor` takes
an `element_surrogate.executor_binding.v1` request containing the exact inspected
`expected_ledger_sha256` and a `rationale`. Adoption of a campaign without observer
context additionally needs its live recorded `legacy_launch_owner` (`pid` and
`created_unix`), observable in this same namespace. All jobs must be quiescent and
guards released. Keep that owner observable until adoption has completed.

Only proven origins are bound; other legacy identities stay unverified and
non-resumable. A verified inactive campaign can hand off only without uncertain
ownership. A proven local reboot books lost active work conservatively. These
local identities are not host authentication for a distributed executor.

## Qualified scope

The rc14 installed Linux suite passed 581 tests, including real separate PID
namespaces, MCP adoption and an actual successor worker. Seventy-nine tests passed
on a VirtualBox shared filesystem. A positive rc13 control reproduced wrong-context
charging of a still-live manufactured worker; rc14 refused the call and preserved
the ledger bytes. Windows had 577 passes, two Linux-only skips and one sandbox
named-pipe refusal; that exact stdio test plus the new recovery test passed when
the local-process restriction was absent. This does not require running normal
agents with administrator privileges.

These are infrastructure and manufactured numerical controls. Fresh scientific
client acceptance remains separately evaluated. No facility field, particle data,
model, private case record or credentials are included in this software.
