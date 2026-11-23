# Durable scientific work

Managed artifacts use per-file OS locks for cooperating readers and publishers.
On filesystems without hard links, exclusive publication checks for any existing
target and renames the complete staged file while holding that lock. Concurrent
external edits that ignore this guard protocol are unsupported. A persistent
`.guard` file is not evidence of a live job: the OS releases its lock on process
exit. No empty destination reservation is used.

Initialize the private project, then configure_scientific_campaign from the
user's supplied budget or authorized delegation. No tool makes that decision.
Do this before heavy numerical operations. Use scientific_job_catalog for the
current allowlist and signatures, and start_scientific_job with an explicit
request. A launch returns a handle; use scientific_job_status rather than
keeping a blocking RPC open or starting a duplicate worker.

The campaign request has schema element_surrogate.scientific_campaign.v1,
limits (threads, working_memory_bytes, scratch_bytes,
max_total_job_wall_seconds, optional max_calendar_seconds), and user_authority
(actor, verbatim user_statement, source_reference). All are explicit. Duration
values can cover hours or days; there is no short production default. Once
recorded, the policy is immutable. Do not evade it by creating another campaign.

The job request has schema element_surrogate.scientific_job.v1, operation,
arguments matching the catalog, resources (threads, working_memory_bytes,
scratch_bytes), max_wall_seconds and time_policy (active or calendar). A calendar
allowance includes queue time/downtime. Active time charges validation, I/O and
computation while a worker owns resources; it excludes queue/downtime. The
campaign charges the sum across jobs, not just the slowest concurrent worker.
Queue admission reserves the worst remaining declared wall allowance, as well
as threads/working arrays/scratch. Observe actual RSS separately: mapped pages,
interpreter and library allocations are additional. Summed process-tree RSS can
double count shared pages. These are cooperating-job controls, not a sandbox
for arbitrary shell code.

Set a realistic allowance from measured representative batches at the actual
integration resolution and settings. summarize_generation_capacity binds timing
receipts to committed shards; it leaves missing timing and untested settings
explicit. Conditional extrapolations do not establish sample sufficiency or a
million-row scale pass. Full startup/source validation/scheduling costs are
additional. Do not reduce the declared scientific survivor target to fit a
chat/pilot duration. Stop/report an inadequate approved allowance faithfully.

The worker runs under a detached supervisor with thread settings applied before
numeric imports. It persists beyond an ordinary client exit on a running OS.
Do not place it under a test namespace whose PID1 is the closing client.
Use the configured persistent MCP service for all job control in one campaign.
On Linux a shell sandbox may use a different PID namespace from that service:
the same integer PID then refers to a different process. Do not mix direct
Python job submission/status with MCP calls across those namespaces. The library
records the executor context before any process probe or ledger mutation and
refuses an unobservable context; a missing PID there is not evidence of death.
Same-context direct Python remains supported. This visibility check is neither
host authentication nor a new scientific resource permission.
Cancellation waits for owned numerical workers to exit before releasing their
permits. Explicit hard deadlines can terminate a bounded batch; only committed
shards count. Inspect scientific_job_status after a reconnect. No duplicate
submission or silent new budget replaces the old job.

For failed/cancelled/incomplete jobs, inspect_scientific_job_recovery verifies
idle ownership and unchanged code/input bytes. resume_scientific_job consumes
that unchanged receipt and remaining original budget. Calendar deadlines never
reset. Interrupted study fits first need inspect_benchmark_recovery and
resume_benchmark_study to preserve/dispose partial fits; in a configured
campaign that prepares recovery without launching another coordinator. Then
inspect/resume the existing scientific job. Only untouched trials may run;
optimizer continuation or retries are not implied. Unknown work after losing
both supervisor and worker is charged conservatively, marked uncertain and
cannot auto-resume. This is not a claim of automatic power-failure recovery.

An exhausted failed diagnostic is different from an interrupted checkpoint.
`inspect_scientific_diagnostic_retry` verifies its frozen request and previously
verified input bytes, inactive owners and released job guard. It returns the
current implementation identity and a request template. Fill a new positive
wall/resource allocation, a new project output root and a reason; submit with
`retry_scientific_diagnostic`. It creates one linked successor inside the original
campaign, keeps every predecessor charge (including uncertain full charges),
retains calendar bounds and explicitly acknowledges any implementation change.
The old attempt earns no passed credit. Identical allocation calls are idempotent;
conflicting forks, changed inputs, stale inspections and insufficient remaining
allowance refuse. This applies to immutable-input diagnostics with fresh derived
outputs; generation, fitting and held-out evaluation keep their own state-specific
recovery protocols. It is not an automatic or unlimited retry.

For an older campaign without executor context, do not edit the ledger or infer
origin from PID numbers. `bind_scientific_executor` requires a request with schema
`element_surrogate.executor_binding.v1`, exact `expected_ledger_sha256`, and a
`rationale`. Legacy adoption additionally requires `legacy_launch_owner` with the
live recorded `pid` and `created_unix`, observable in this same namespace. All
jobs must be quiescent and guards released. Only identities whose origin is proven
are bound; others remain unverified and cannot be resumed. A quiescent verified
campaign can hand off to another context only with no uncertain owners. A proven
local reboot is handled conservatively without refunding work. This is a local
cooperating-executor protocol; caller attestations do not authenticate a human.

Before launching long jobs, run the full dependency doctor with the exact Python
configured for this project's MCP server, from the actual client isolation.
A doctor run in an administrator shell can see different user-site dependencies
and therefore does not qualify that isolated runtime.

Generation's blocking internal operation accepts max_batches as an optional
scheduling boundary. It never changes the frozen total target or RNG identity.
build_training_data_request takes explicit scheduling/count decisions, computes
the transport payload and preserves exact qualified settings/ranges. It returns
the resulting plan hash for separately supplied bound user attestations. Do not
handwrite an approximate Transport.description() or silently alter a setting.

The package checks explicit saved/backend numerical thread counts against a
managed reservation, including later loading/export. Public ONNX sessions use
the model's recorded thread count instead of the machine-wide runtime default.
Managed PyTorch inter-op work is sequential. A larger stored count needs a job
with sufficient already-authorized resources; never weaken the guard or alter
checkpoint metadata to hide it. Auxiliary I/O/runtime threads are distinct from
parallel numerical workers; reservations do not claim an OS thread-count cap.


## Pilot to full generation

Fault recovery and planned continuation are different operations. Recovery repeats
an interrupted execution with unchanged arguments and only its remaining original
allocation. After a clean max_batches pause, inspect_scientific_job_recovery and
continue_training_data_phase can link a new execution allocation to that exact
checkpoint. The continuation request is:

```json
{
  "schema": "element_surrogate.generation_continuation.v1",
  "max_batches": null,
  "max_wall_seconds": 86400,
  "rationale": "Continue the measured pilot within the already authorized full plan"
}
```

The duration above illustrates syntax, not a default or new permission. Choose a
measured allocation within the original plan and campaign allowance. Null means
all remaining batches, not an unbounded scientific target. Optional resources
can change reservations within the same original campaign limits. Inputs,
qualification, settings, ranges, RNG seed, batch size, approvals and committed
rows do not change. Prior phase costs remain charged; no original calendar
boundary is extended. Only one successor is allowed. A repeated identical call
returns that successor, and the old phase cannot resume alongside it. Faults,
uncertain accounting or dirty checkpoints use recovery, not this operation.

Status handles expose is_pending and is_terminal. Queued and starting jobs are
pending; neither is a failure. A terminal incomplete phase still needs an
explicit continuation or recovery decision. Preserve immutable operation-specific
request filenames while jobs are queued/running. Order dependent study reporting
and export calls after the study owner releases its lock; transient contention
is not permission to remove that lock or refit.
