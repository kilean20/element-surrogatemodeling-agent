# Shipped knowledge base

One Markdown file per entry, with YAML front matter, grouped by kind in folders
(`methods/`, `simulators/`, and later `magnets/`, `lessons/`). The file name equals the
entry's `id`. `element_surrogate.knowledge` loads and validates the entries, and the
tools `search_knowledge` and `read_knowledge` serve them.

## Front matter

| field | meaning |
|---|---|
| `id` | the file name without `.md` |
| `title`, `summary` | what the entry is about, in words a search would use |
| `kind` | `method`, `magnet_class_profile`, `simulator_profile`, `case_record`, `lesson` or `reference` |
| `status` | `proposed` (awaiting an applicable review), `active`, `superseded` or `retired` |
| `date`, `replaces` | when it was written and the entry it supersedes, if any |
| `approved_by` | empty while proposed; `user`, or `agent-approved` when the user lets the agent approve its own entries |
| `keywords` | plain search terms |
| `evidence` | each item's `source`, what it `supports`, and whether that support was `verified` |
| `tools`, `tests` | the registered tools and the package tests that implement or check the entry |

## Rules for shipped entries

- Only what holds for a magnet the agent has never seen: physics, definitions,
  conventions, simulator behaviour, procedures and pitfalls.
- Evidence is a public source, a derivation in the entry, or a test in this package.
  No facility field maps, case records, file paths or numbers from one facility's magnets.
- A method entry describes a computation; the computation itself is a tested tool,
  which the entry names (or marks as planned).
- No distance or threshold from one magnet becomes a rule for others; tolerances come
  from each case's own measured errors.
- New shared entries remain proposed until maintainer review. Private entries
  become active through explicit user disposition or validated, recorded local
  delegation, with honest `user` or `agent-approved` provenance. This does not
  modify shared knowledge or grant sharing/scientific gate authority.
