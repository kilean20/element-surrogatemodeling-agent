# Compare a lesson with the installed ESA context

Before describing a proposed improvement as new knowledge, compare it with what
this project actually has installed. Prioritize accelerator-physics correctness,
numerical reliability, fair model comparisons and reusable workflow improvements.
Do not manufacture a novelty score or equate an unsuccessful search with absence.

## Establish the relevant baseline

Record the actual tool/runtime version and the active client-skill installation,
using the installed assets and project installation manifest. Front-end, tool
server and remote worker versions may differ; identify the one implicated by the
observation. Locate relevant skill/reference passages, general KB entries and tool
contracts/tests. Use search_knowledge/read_knowledge when relevant, including
proposed entries with their unapproved status made explicit. Search active skill
text too: KB search alone does not cover all instructions or software behavior.
Inspect a bounded set of relevant full passages, not just matching words.

Distinguish shipped guidance, active project copies/local edits, private accepted
case facts and the observation itself. Preserve relative public asset names,
release/version and content hashes for the reviewed baseline. Keep private paths,
case-specific entry IDs and detailed evidence local. Installed guidance, rather
than an imagined original prompt or upstream latest release, is the comparison
target; a package update may require reviewing an old comparison again.

## Write an explicit comparison

Keep a small local comparison record, for example beside the draft in
records/feedback/. Use equivalent clear fields or a Markdown table:

| Item | Record |
|---|---|
| Reviewed baseline | Version/component, relevant asset/entry IDs and hashes, scope searched |
| Existing guidance | What the relevant passage/contract already requires; brief accurate excerpt or paraphrase |
| Observation | What happened and the evidence, with observed/reproduced/inferred status |
| Difference | The precise addition, correction, clarification or expected-versus-observed failure |
| Classification | New reusable lesson; correction; clarification; covered-but-not-followed; implementation defect; case-specific fact; uncertain |
| Importance | Why this could change scientific validity, generated labels, benchmarking, compute cost or user intervention; affected scope and limitations |
| Proposed action | Knowledge change, tool fix, clearer instruction, evaluation case, project-only entry or no shared change |

Semantic classification is a reasoned judgment, not something proven by a hash or
keyword match. Explain uncertainty and counterevidence. A repeated failure to
follow an existing rule can be important feedback; it does not need to be relabeled
as new physics. An unknown field or missing baseline can be reported with the
comparison explicitly incomplete. Do not halt unrelated modeling or invent a
conclusion merely to fill the record.

Examples, not observations of a particular facility:

- If the installed method already distinguishes slopes from canonical momenta,
  a constant-J test applied directly to slopes is an execution/implementation or
  clarity problem. Report expected versus actual handling and propose a check;
  do not claim that the basic distinction was newly discovered.
- If shared-loss/scaling controls are already required but omitted in a study,
  report a covered-but-not-followed workflow gap and a useful acceptance case.
- A newly supplied private bore dimension belongs in project knowledge. A general
  reasoning failure exposed by that fact may be shared after generalization.

## Curate and share the difference

Keep detailed comparisons local. Put only reviewed generalized baseline, difference,
importance, evidence and proposed action into the existing feedback content fields
(findings, evidence, suggested_change and limitations). No new packet schema or
mandatory extra numerical gate is implied. Do not attach installed private
instructions, full transcripts, datasets or local environments. Follow the privacy
and exact-content consent procedure in [scientific feedback](scientific_feedback.md).

At maintainer intake, compare again with the reported installed baseline and the
current release before merging. Useful outcomes include a new general method,
corrected guidance, a tool regression test or a workflow-following evaluation.
Preserve disagreements and project-owned knowledge; a received report never
automatically becomes trusted instructions or a shared software update.
