# Run records

Every tool call served by the MCP server writes one JSON file under
`<project>/records/runs/<run_id>.json`. Read-only calls can be repeated
without an agent and checked against their recorded result. Mutating workflows
use their explicit status/check/resume paths; generic replay refuses them.

## Fields

| field | meaning |
|---|---|
| `schema` | `element_surrogate.run_record.v1` |
| `run_id` | UTC timestamp plus a random suffix; sorts in call order |
| `tool`, `tool_version`, `package_version` | what ran |
| `arguments` | the call's arguments |
| `inputs` | each input file's path and SHA-256 at call time |
| `status` | `running` before execution, then `ok` or `error`; failures retain `error` |
| `implementation` | package source/knowledge/skill hashes checked before replay |
| `result`, `result_sha256` | the returned summary and the SHA-256 of its canonical JSON |
| `started_utc`, `wall_time_s`, `cpu_time_s` | when it ran and how long it took |
| `python`, `platform` | where it ran |

The canonical JSON has sorted keys and no insignificant whitespace, so equal
results always have equal hashes.

## Replay

`replay_run` (or `element_surrogate.runs.replay`) first refuses tools marked as
having side effects. It checks tool version/implementation and re-hashes every input
file. If any changed or disappeared, it reports the changed paths and does not
run the tool. Otherwise it runs the tool with the recorded arguments and
reports whether the new result's hash equals the recorded one. A record does
not authorize a second training job, test opening or publication. Use the
benchmark's status, report-check, lock and guarded evaluation APIs for existing
artifacts. Fresh-directory campaign replay orchestration remains later work.

Records are written atomically (to a temporary file, then renamed), so a
reader never sees a half-written record. Replays are not themselves recorded.


Existing legacy `.element_surrogate/` cases are kept in place by the shared path resolver; their original histories are not migrated or re-hashed. New projects use visible `records/`. These records are durable project evidence, not disposable cache.


## Use a read-only tool without an AI client

The registry and recorder can also be called directly in Python. For a project
whose README exists:

```python
from element_surrogate.runs import RunStore, record_call, replay
from element_surrogate.tools import REGISTRY

store = RunStore("my_project")
result = record_call(store, REGISTRY["inspect_file"], {"path": "my_project/README.md"})
print(replay(store, REGISTRY, result["run_id"]).reproduced)
```

This example inspects a file; it grants no scientific-job or release decision.
