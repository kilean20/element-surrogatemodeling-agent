# Starting with supplied training labels

Reuse suitable labels when the user supplies them. Do not regenerate a training
dataset merely to obtain a generation ledger. Assess the source field and the
label interpretation independently; importing bytes does not certify physics.

1. Inspect the supplied coordinates, units, native planes, reference/charge,
   control semantics, splits/IDs, source identities, numerical settings and prior
   test use. Convert units only through a recorded transformation with new
   identities. The current native adapter expects SI states m/rad and its
   documented 5D/6D momentum/arrival convention. Unknowns remain unknown.
2. Read `element-surrogate request-example supplied-study`. It contains a full
   benchmark protocol and `label_provenance` with `qualification_status:
   source_declared` and explicit train/validation/test history. Supply actual
   cohort file hashes, dataset identity, measured physics evidence, resource
   decisions and candidate configurations. The example is a template with
   placeholders, not an executable scientific case. `source_contract` describes
   the producer's transport/ranges/settings for card and bounded native API
   interpretation; it is not a qualification made by this importer. Supply an
   explicit coupling interval only if that continuous scope is declared;
   discrete settings alone do not establish the intervening interval.
   Structural candidates need current measured evidence bound to the declared
   transport and supplied source envelope, just as for generated-label studies.
   This does not rewrite the older labels' producer qualification or provenance.
3. Call `prepare_supplied_neural_surrogate_study(request_path, output_dir)`.
   It checks train/validation metadata, IDs and static momentum conservation,
   binds the real adapter, and preserves all existing splits. It hashes test
   bytes for identity without reading test targets. It creates no labels or
   physics approval. Never invent `generation_record_path` for supplied data.
4. If useful, run `assess_supplied_label_reference(transport_path, protocol_path,
   request_path, output_path)` on an explicit seeded train/validation sample
   with native component limits. Read `request-example supplied-label-check`.
   Use a separately numerically qualified transport; agreement with a same-code
   label source is consistency evidence, not independent physical proof. This
   finite diagnostic is not a new training campaign. Report unsupported rows.
5. Complete the same physics review, bounded benchmark, selection lock, export
   and full inference-cost checks as for generated data. The common workflow
   has no fixed row cap. Prior-used/unknown test rows support a historical
   regression, not a newly untouched holdout. Restrict any fresh evaluation claim
   to a truly new declared evaluation cohort sealed until selection.
6. Call `prepare_supplied_genesis_release_cards(study_path, transport_path,
   label_dataset_dir, field_dataset_dir, request_path, output_dir)`. This uses
   the same pinned Genesis templates/introspection/validators as generated
   cards. It binds source declarations and cohort identities without fabricating
   a local generation record. Keep the field and particle cards separate and
   retain source-declared qualification, prior use and missing facts. Complete
   technical sections after jobs finish; human rights/ownership remain the
   current user's decisions. No host Genesis skill is required.

Keep immutable originals and their exact derived mapped representations together
under `inputs/fields/` (a separate subdirectory per representation is useful).
The field dataset directory given to card tools must contain the representation
actually named by the transport, with original-to-derived hashes and conversion
provenance retained. Diagnostic requests/results belong in `studies/` or
`reports/`. A directory containing only the original file cannot describe a
mapped transport stored elsewhere merely by naming the same magnet.

Export resolves the API envelope from the hash-bound source declaration. It
retains source-declared provenance, supported-trajectory conditions and prior
test history. If that declaration is absent, the study can remain useful, but a
bounded native API export needs the missing ranges/settings before completion.

Heavy import diagnostics, card introspection and benchmarking use the ordinary
campaign/job layer. Client timeouts are not scientific deadlines. Existing
generation/recovery tools remain the route for locally generated cohorts.
