# Preparing ESA for GitHub publication

The reviewed source folder is the software distribution. Users can download a
tagged GitHub release's **Source code (zip)**, extract it and follow
[USER_PROMPTS.md](USER_PROMPTS.md). GitHub generates that download; no custom ZIP,
bundled wheel or environment is needed. A clone works too. See
[GitHub's release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Publish reviewed software

Use a clean software copy, preserving the source tree and hidden source metadata.
Keep the release identity consistent in `pyproject.toml`, the runtime,
[CHANGELOG.md](CHANGELOG.md), [QUALIFICATION.md](QUALIFICATION.md) and release notes.
The prepared [release-note draft](docs/release_notes.md) describes the current
research-preview scope.

Include root instructions/helper, licenses and notices, `pyproject.toml`,
`uv.lock`, `MANIFEST.in`, source, skills, docs, cards, tests and normal developer
metadata such as `.github/`, `.gitignore` and `.gitattributes`. Keep their relative
paths. Users need no software from the maintainer's research projects.

Keep these out of committed source and downloads:

- Python environments and downloaded dependencies;
- caches, bytecode, build/packaging outputs, coverage and editor state;
- credentials, tokens, private keys, `.env` files and client authentication;
- initialized `.agents/`, `.codex/`, `.claude/` and `.mcp.json` user configuration;
- user `inputs/`, `data/`, `studies/`, `models/`, `reports/`, `records/`, legacy
  case state and equivalent scientific artifacts stored elsewhere;
- facility fields, particle cohorts, magnet checkpoints and private case memory.

These exclusions do not authorize deleting user work or qualification evidence.
Keep them separately. `.gitignore` reduces accidental additions; review the actual
selected files. Starting from a clean copy also avoids importing an unrelated
workspace's history. Git's own `.git/` database is local repository metadata and
is not part of source downloads; the clean distribution does not bundle it.

## Preserve and verify release identities

`.gitattributes` preserves the byte-level source identities used for installation
and pinned-resource checks, including across Windows/Linux Git settings.
`SHA256SUMS`, when supplied, lists software files relative to this folder and
excludes itself. Regenerate it after any intended content change; it is a
consistency record, not a publisher signature or a privacy guarantee.

Before publication, check internal links, the selected files and checksums,
runtime/skill identities and the relevant executed tests. Preserve old evidence
with its actual version/scope. A configured CI workflow is not an executed CI
result; do not add a passing badge until the corresponding run succeeds.

When the owner chooses to publish, create the repository/release under the
owner's authority, tag the reviewed revision and use the draft release notes.
Send first users the **release-page link** and point them to README and
USER_PROMPTS. GitHub creates source downloads from the tag. If publishing before
a release exists, **Code → Download ZIP** downloads the selected branch; a tagged
release makes the tested version easier to identify.

Publication preparation does not itself create a remote, commit, tag, push,
release or archive. Those are separate authorized actions.

The Apache-2.0 software license and third-party notices remain in force. They do
not assign ownership or licensing to a user's future software, fields, generated
particles or models. No facility-data approval is invented for software-only
distribution; any actual established obligations still apply.
