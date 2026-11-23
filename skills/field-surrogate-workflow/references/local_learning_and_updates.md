# Local learning and maintainer updates

Use this procedure when a case produces a useful lesson or the user wants to
adopt a newer ESA. Neither path requires Drive. Keep the user's project
knowledge and the installed software as separate layers.

## Learn within the user's case

1. Restore the case. Compare the observation with the actual installed guidance,
   active project knowledge and recorded user decisions; use
   `inspect_project_learning_context` and the
   [installed-guidance comparison](installed_guidance_comparison.md).
   State the new lesson, its evidence, scope and limits. A failure to follow an
   existing instruction is different from missing knowledge.
2. Use `propose_project_knowledge` for a distinct private entry in
   `records/knowledge/`. A method may refer to a versioned local script and its
   checks; creating or running that script uses the harness's normal permissions.
   This tool does not patch the installed package or shared KB.
3. The default is manual activation through
   `record_project_knowledge_decision`, using the user's actual decision.
   For repeated local learning, offer a narrow, revocable delegation: knowledge
   kinds, scope and whether replacements are allowed. Use
   `element-surrogate request-example learning-policy --output <new-path>`
   to draft it, then `record_project_learning_policy` with the reviewed file
   hash, user's words and their source. Never invent consent.
4. Under an active delegation, validate before using
   `activate_validated_project_knowledge`. Draft the validation with
   `request-example learning-validation`. Record the current entry/context
   hashes, comparison with installed guidance, scope assessment, conflict review
   and local evidence files/hashes with what each supports. Validation is
   change-directed: an analytic check, a documented source/user fact, regression
   probes or a controlled numerical study may be appropriate. Do not fabricate a
   fit to approve a documentation lesson.
5. Supported, in-scope activation is recorded as `agent-approved`, not
   `user`. Inconclusive or stale evidence retains a proposal; unrelated work
   can continue. Explicit user-accepted knowledge cannot be replaced through
   delegated activation. Resolve that conflict with the user.
6. Revoke delegation by recording a new policy with `mode: manual`.
   Policy history remains. Revocation stops future delegated activations; inspect
   and explicitly disposition any previously accepted entries needing retirement.

These records bind explicit files and attestations, not human identity or
scientific truth. Scope compliance and useful validation still require judgment.
Delegation does not approve scientific gates, expensive work, publication,
external sharing or changes to the shared KB. Search tools distinguish proposed
from active knowledge; do not treat a proposal as an accepted premise.

## Adopt a newer ESA without replacing the case

Do not rerun initialization to overwrite local edits. Keep the current Python
environment available. Install the candidate into a separate environment and
run its doctor; then use that candidate's CLI or MCP tools. No running process
is upgraded in place.

1. Reach an idle checkpoint for the client/runtime being switched. Save jobs and
   their handles; do not kill long scientific work to install an update. Arrange a
   later switch if jobs still need the current runtime.
2. `stage_project_client_update` records the current managed files, incoming
   files and prior vendor baselines. It stages only client skills and the ESA
   MCP connection. Fields, labels, models, private knowledge, user decisions
   and compute accounting are not update targets.
3. `inspect_project_client_update` returns the plan/hash, file conflicts,
   shared-KB changes and an unapproved review template. Compare actual diffs,
   release notes, active local methods and affected interfaces. Three-way
   classification preserves a deliberate local variant when the vendor copy
   has not changed; simultaneous changes require a choice.
4. Resolve each file conflict explicitly with
   `resolve_project_client_update`: `keep_local`, `candidate` or
   `merged`, with a rationale. A merged file must be inside the case.
   Inspect the final selected files and links. Each resolution changes the plan
   hash. Keep the new hash for review and application.
5. Save the review inside the case. Its three checks are
   `guidance_and_local_knowledge`, `runtime_and_interfaces`, and
   `affected_project_behavior`. Each needs an outcome
   (`passed` or `not_applicable`), rationale and local evidence paths/hashes.
   A diff plus inspection record can justify a behavior check being inapplicable
   to a prose-only update. Changed numerical behavior needs relevant probes or
   studies. No automatic full retraining is required for every update.
6. `apply_project_client_update` requires the current plan hash, unchanged
   context/runtime/evidence and an attested idle checkpoint. It verifies recovery
   assets before writing and journals progress. Reconnect the client and review
   its tool/skill discovery using the new environment.
7. If adoption or recovery needs review, keep supported work on the pinned old
   runtime available. Missing legacy baselines, shared/private ID collisions,
   unavailable recovery bytes or later user edits need explicit reconciliation.
   Do not delete or silently rename private knowledge to force adoption.
   After restoring a pending recovery asset, retry the inspected recovery path.

CLI equivalents, run with the candidate interpreter:

```text
python -m element_surrogate.cli update stage --project <case> --id <update-id>
python -m element_surrogate.cli update inspect --project <case> --id <update-id>
python -m element_surrogate.cli update resolve --project <case> --id <update-id> --plan-sha256 <current-hash> --path <managed-path> --choice keep_local --rationale "<reason>"
python -m element_surrogate.cli update apply --project <case> --id <update-id> --plan-sha256 <reviewed-hash> --review <case-relative-review.json>
python -m element_surrogate.cli update rollback --project <case> --id <update-id>
```

`request-example update-review` is a draft, not a passed review.
`inspect` supplies this plan's hashes. Rollback preflights all recovery bytes,
preserves later edits and records a retryable journal on write failure. It does
not roll back scientific work or package installations. Keep the old environment
until the new setup is accepted; point the client back to the recorded prior
setup after rollback.

Per-file publication and journals reduce interruption risk; they do not provide
a filesystem-wide atomic transaction, semantic merge or compatibility proof.
A later change to knowledge/configuration can invalidate the original rollback
context. Preserve it and stage a reviewed reverse update or reconcile manually,
rather than discarding subsequent user work.

## Optional shared feedback

Only curate general lessons after comparing installed guidance. Keep machine
identities, source data and sensitive specifics local. Local learning is useful
even when nothing is shared. Follow the
[Drive access procedure](google_drive_feedback_access.md) and exact-payload
sharing consent separately. Sharing a lesson neither activates it in another
user's case nor automatically changes a future ESA release.
