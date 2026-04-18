---
name: workspace-init-skills
description: "Initialize a project workspace with a .claude directory containing Anthropic's skills repo and PyTorch upstream Claude skills. Use when starting a new project, onboarding into a workspace that lacks a .claude folder, or when the user mentions initializing .claude, setting up Claude skills, or workspace initialization."
---

# Workspace .claude Initialization

Bootstrap a project's `.claude` directory by cloning [anthropics/skills](https://github.com/anthropics/skills) and merging [pytorch/pytorch .claude/skills](https://github.com/pytorch/pytorch/tree/main/.claude/skills) into it.

## When to Use
- Starting work in a project that has **no `.claude` folder**
- User asks to "initialize .claude", "set up Claude skills", or "bootstrap workspace"
- Beginning of any new project onboarding

## Precondition Check

Before doing anything, check if `.claude` already exists at the workspace root:

```bash
if [[ -d ".claude" ]]; then
  echo ".claude directory already exists — skipping initialization."
fi
```

**If `.claude` exists, STOP. Do not re-clone or overwrite.** Inform the user and ask if they want to update instead.

---

## Procedure

### Step 1 — Clone anthropics/skills as .claude

Clone the Anthropic skills repository into the project root, naming the directory `.claude`:

```bash
git clone https://github.com/anthropics/skills.git .claude
```

If behind a proxy, ensure `http_proxy` / `https_proxy` are set first (check `env.sh` if available).

After cloning, **remove `.git`** so `.claude` is a plain directory (no upstream tracking):

```bash
rm -rf .claude/.git
```

**Verify:**
```bash
ls .claude/SKILL.md 2>/dev/null || ls .claude/template/SKILL.md
```

### Step 2 — Sparse-checkout PyTorch .claude/skills

Clone only the `.claude/skills` subtree from upstream `pytorch/pytorch` into a temporary directory, then move the contents into `.claude/skills/`:

```bash
TMPDIR=$(mktemp -d)
git clone --depth 1 --filter=blob:none --sparse https://github.com/pytorch/pytorch.git "$TMPDIR"
cd "$TMPDIR"
git sparse-checkout set .claude/skills
cd -
```

### Step 3 — Merge PyTorch skills into .claude/skills

Move (or copy) all PyTorch skill folders into the `.claude/skills/` directory. Avoid overwriting any existing skills from Anthropic's repo:

```bash
# Ensure target directory exists
mkdir -p .claude/skills

# Move each PyTorch skill folder, skip if name conflict
for skill_dir in "$TMPDIR/.claude/skills"/*/; do
  skill_name=$(basename "$skill_dir")
  if [[ -d ".claude/skills/$skill_name" ]]; then
    echo "SKIP: .claude/skills/$skill_name already exists (from anthropics/skills)"
  else
    mv "$skill_dir" ".claude/skills/$skill_name"
    echo "ADDED: .claude/skills/$skill_name"
  fi
done
```

### Step 4 — Cleanup

Remove the temporary PyTorch sparse checkout:

```bash
rm -rf "$TMPDIR"
```

### Step 5 — Verify

List the final contents and confirm both Anthropic and PyTorch skills are present:

```bash
echo "=== .claude/skills contents ==="
ls .claude/skills/
echo "=== Total skill count ==="
ls -d .claude/skills/*/ | wc -l
```

Expected PyTorch skills include: `add-uint-support`, `aoti-debug`, `at-dispatch-v2`, `docstring`, `document-public-apis`, `metal-kernel`, `pr-review`, `pt2-bug-basher`, `pyrefly-type-coverage`, `scrub-issue`, `skill-writer`, `triaging-issues`.

---

## Post-Initialization

After setup, inform the user:
1. `.claude` directory is ready with both Anthropic and PyTorch community skills.
2. They can add project-specific skills to `.claude/skills/` following the template in `.claude/template/SKILL.md`.
3. Consider adding `.claude` to `.gitignore` if they don't want to commit it.

## Error Handling

| Problem | Resolution |
|---------|-----------|
| `git clone` fails (network) | Check proxy settings; retry with `GIT_SSL_NO_VERIFY=true` if on corporate network |
| `.claude` already exists | Skip entirely; ask user if they want to update |
| Sparse checkout fails | Fall back to full clone with `--depth 1`, then `cp -r` just the `.claude/skills` subtree |
| Name conflict in skills merge | Skip the conflicting folder, log which ones were skipped |
