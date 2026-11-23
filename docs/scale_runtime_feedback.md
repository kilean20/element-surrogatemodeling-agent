# Planning data scale, runtime and improvement feedback

The installed workflow contains the full [procedure](../skills/field-surrogate-workflow/references/population_runtime_feedback.md)
and [compute connection checklist](../skills/field-surrogate-workflow/references/compute_connections.md).
These work without the developer's project, field data, memory or global Genesis skills.

```text
User goal and limits
  -> explicit survivor target/splits/coverage
  -> representative measured pilots
  -> generation ETA + model-study ETA + missing/queue/transfer costs
  -> >12 hours? ask about faster compute or continue locally
  -> durable generation, actual count verification
  -> data-size/convergence checks + comparable-cost benchmark + final fit
```

Useful first prompt:

> Build a full 4D research surrogate from the field in inputs/fields. Generate at
> least 5 million total surviving particles after all validity filters. Propose
> splits, coverage and a model study; estimate generation and study time from
> measured pilots. Ask me about faster compute if this is likely to exceed half
> a day. Keep the test held out and report actual fitted as well as generated counts.

The five-million value belongs to that user's request, not a factory default for
all magnets. Existing supplied-data studies still use supplied labels; no ledger
requirement forces costly regeneration. A full study with delegated N must propose
a scientifically justified target and test adequacy instead of silently using a
tiny demonstrator. Neither count nor a fixed update cap certifies convergence.

Public tools: build_training_data_request (optional population_policy_path),
assess_population_requirement, summarize_generation_capacity and
estimate_workload_time. Installed examples include population_policy.json,
workload_request.json and workload_observation.json. The timing examples are
structural examples: fill measurements from real pilots and supply their source
receipts. They cannot establish performance on the user's computer.

```text
Local user feedback / observed lesson
  -> curated report in records/feedback (nothing sent)
  -> user reviews exact text + recipient + destination + hash
  -> approve / decline / revoke
  -> optional authorized upload OR manual reviewed file
  -> maintainer inbox: review -> reproduce -> test -> versioned release
```

Tools: prepare_improvement_feedback, record_feedback_consent,
send_improvement_feedback and collect_improvement_feedback. Installed
feedback_request.json prefills the owner-requested interim ESA-improvement
Drive recipient/folder. Replace example content with generalized scientific
lessons and review the exact destination; it grants no enrollment or consent.
See [curation and privacy](scientific_feedback.md).
No background collection or upload runs by default. Do not put credentials in a
prompt, report or tool argument; transport takes an environment variable NAME.
A token configured elsewhere still needs authorized access to the specific folder.
Local consent is a client attestation, not authenticated identity. A packet can
contain sensitive text if curated badly. Content-only preflight flags common
credentials, paths and identifiers and refuses sharing until redrafted; human
semantic review remains essential. No automatic confidentiality guarantee is claimed. Revocation is prospective. A timeout/crash after
upload submission requires checking the remote packet hash before another attempt.
Google Drive is optional; no SDK/global skill is required, and live permissions
are not certified by offline transport tests. The source field/data/model ownership
and sharing terms remain the future user's decisions.
