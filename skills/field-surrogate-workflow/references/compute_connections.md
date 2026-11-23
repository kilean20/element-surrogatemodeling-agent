# Optional faster compute: connection and migration checklist

This standalone reference follows the reviewed Genesis AmSC Python-client
connection procedure and primary facility documentation. It does not require a
globally installed Genesis skill, an AmSC account, a particular cloud or a
maintainer's connection. Available compute can be a local GPU, user server or an
allocated cluster. Reuse a user's established authorized connection when suitable.

1. Ask what is actually available: CPU/GPU type/count, memory, storage/quota,
   interface (SSH/Jupyter/AmSC), facility, account/allocation and queue limits.
   The user owns resource and cost decisions. A speedup proposal is not job or
   data-transfer authorization. Check whether project data may leave its current
   location; apply the user's stated handling/ownership terms.
2. Authenticate through the user's own approved client/terminal/browser. Do not
   request passwords/tokens in conversation, copy another profile's sign-in or
   put credentials in case logs/cards/feedback. For AmSC, use the SDK's current
   documented auth preflight before API calls; missing access is not a reason to
   invent a token. For SSH use the site's approved login/MFA route. Do not assume
   the developer's localhost address resolves to the user's compute endpoint.
3. Read the resource's scheduler/access policy and installed runtime. Use
   Slurm/PBS or another supplied scheduler correctly: account, partition/queue,
   walltime, CPU/GPU allocation, filesystem and actual launch/binding. Do not run
   large production fits on a login node. Containers are optional, not universal.
4. Preserve software/release/source hashes, coordinate/reference/field contracts,
   IDs, filters, splits, budgets, selection locks and test history on migration.
   Transfer only authorized necessary artifacts with destination identity and
   sufficient disk/memory. Never reopen test data merely because hardware changed.
5. Run a small real pilot there. Verify numerical agreement within the declared
   tolerances and measure actual throughput/peak memory. Include queue, transfer,
   setup and checkpoint/I/O costs. Update generation and study estimates before
   scheduling the remainder; GPU availability alone does not imply a speedup.
6. Use durable jobs, explicit submission IDs, status/cancel/resume protocols,
   committed batches and checkpoint records. Distinguish queued/running/failed/
   completed from a disconnected client. On restart inspect original job/charges
   before resubmission to avoid duplicate workers. Report actual user decisions
   and any deviations; preserve the local option when no faster resource exists.

Connection details change. Consult primary documentation for the actual site:
[NERSC connections](https://docs.nersc.gov/connect/) and
[ALCF user guides](https://docs.alcf.anl.gov/). Other sites use their own official
instructions. Optional installed AmSC/IRI/Slurm/PBS skills can add site-specific
help, but these are not hidden dependencies of the element workflow. This guide
does not qualify live access, job submission or scientific performance on any site.
