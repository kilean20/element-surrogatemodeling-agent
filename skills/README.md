# Skills

Start with [field-surrogate-workflow](field-surrogate-workflow/SKILL.md).
`field-map-readiness`, `surrogate-review` and `surrogate-cards` cover the
corresponding scientific and release steps. These are local client procedures;
full autonomous/client acceptance remains separate.

The [surrogate-benchmark](surrogate-benchmark/SKILL.md) skill guides a bounded
scientific model study through the package's tested MCP tools. Its protocol
reference documents the adapter, units, selection and approval contracts.

Skills follow the layout AI-ModCon's
[genesis-skills](https://github.com/AI-ModCon/genesis-skills) validates,
with one named directory containing `SKILL.md`; domain grouping can be added
when there are more skills. The wheel installs these skills under
`share/element-surrogate/skills/`; the source distribution includes them here.
No host-wide skill installation is performed automatically.

Each skill follows the [Agent Skills specification](https://agentskills.io/specification):

- `name`: 1 to 64 lowercase letters, digits and single hyphens, equal to the
  directory name;
- `description`: at most 1,024 characters, saying what the skill does and when to
  use it;
- `SKILL.md` under 500 lines, with detail in `references/` and code in `scripts/`;
- paths relative to the skill root, with no hard-coded client directories;
- an attribution file, and optionally `agents/openai.yaml` for Codex.

CI validates every skill with genesis-skills' `tools/validate_skills.py
--profile all` at a pinned commit. Skills meant for upstream are developed and
exemplified on synthetic fields only.
