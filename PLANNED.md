# Planned Features

This document outlines planned work for the `cadalloc` crate. It is a design
surface, not a backlog: an entry exists so the decision can be argued with
*before* anything is built, and it should carry enough reasoning that a reader
who disagrees knows exactly which claim to attack.

Entries aim to be backward-compatible. New capability is preferably added under
a feature flag and behind new traits rather than by modifying existing ones.

A shipped entry moves to `CHANGELOG.md` under `[Unreleased]` and is deleted
from here.

---

## Entry format

Each entry is a `##` heading, followed by a metadata block, followed by three
required subsections.

```markdown
## `feature_name` — one-line description (target version)

**Feature flag:** `flagname` (implies `other`), or `None`.
**Breaking change:** No — new inherent methods only.

### Motivation
### Design
### Open questions
```

**Metadata** is per-crate; pick the axes that carry real consequences for this
one. `Feature flag` and `Breaking change` apply almost everywhere. Others worth
considering: `Category`, `Depends on` (another entry, by name), `Target
version`, `On-disk format`, `MSRV`. State `No` explicitly rather than omitting a
field — an absent `Breaking change` line reads as "not yet considered".

**Motivation** — the problem, concretely, ideally with the call site that is
awkward today. What does a caller have to write now, and why is that bad? An
entry whose motivation is "it would be nice to have" is not ready.

**Design** — the proposed shape. Signatures, a code block, the invariants it
maintains, and what it explicitly does *not* change. Long entries subdivide with
`####`.

**Open questions** — what is genuinely undecided, as a bulleted list with a bold
lead-in per item. This section is not optional and "none" is a suspicious
answer; an entry with no open questions is usually one that has not been
examined closely enough to have found them.
