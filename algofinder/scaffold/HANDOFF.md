# Handoff: finish the venv shebang fix (algofinder package rename)

## Context

The Python package inside `projects/stem4humanity/algofinder/` was renamed
from `stem4humanity` to `algofinder`:

- `algofinder/stem4humanity/` -> `algofinder/algofinder/` (via `git mv`, rename is staged in the index)
- All references updated (imports, `python -m` commands, pyproject.toml, READMEs, docs)
- Verified end-to-end BEFORE this remaining task:
  - `generate` (8 problems, manifests byte-identical), `train` (7 ML models),
    `benchmark` (496 runs, 0 errors), `report` (leaderboard rendered)
  - `import stem4humanity` fails; `import algofinder` works

## Your task

Fix the pre-existing broken console scripts in the project venv
(`algofinder/.venv`). Their shebangs point at the old repository path (the
repo root was renamed to lowercase `stem4humanity`, breaking every entry-point
script). The venv itself works and its Python executable has all dependencies.

### Broken scripts (text files, confirmed by scan)

These 11 files in `.venv/bin/` contain the old path and must be fixed:

```
activate  activate.csh  activate.fish  f2py  numpy-config
pip  pip3  pip3.13  pygmentize  py.test  pytest
```

Fix: define the old and new roots, then replace the old path prefix with the
new one in those files only (non-destructive content edit; `sed` is fine for
this bulk shebang rewrite):

```bash
OLD_PROJECT_ROOT=/path/to/projects/STEM4Humanity
PROJECT_ROOT=/path/to/projects/stem4humanity
sed -i "s|${OLD_PROJECT_ROOT}|${PROJECT_ROOT}|g" \
  .venv/bin/activate .venv/bin/activate.csh .venv/bin/activate.fish \
  .venv/bin/f2py .venv/bin/numpy-config .venv/bin/pip .venv/bin/pip3 \
  .venv/bin/pip3.13 .venv/bin/pygmentize .venv/bin/py.test .venv/bin/pytest
```

Do NOT recreate the venv (`python -m venv --clear ...` deletes files — forbidden).
Do NOT touch `.venv/bin/python*` symlinks (they are correct).

### Verify

```bash
.venv/bin/pip --version
.venv/bin/python3 -m pip --version
.venv/bin/pytest --version
.venv/bin/python3 -m algofinder.harness.runner --config config/config.toml generate
```

Use `.venv/bin/python3 -m ...` if any script still fails; report remaining breaks.

## Leave alone (intentional / inert)

- `stem4humanity` references kept on purpose (do NOT "fix"):
  - `algofinder/algofinder/trace/recorder.py:11,44` + `docs/dev-mode-observability-design.md:428,800` —
    trace envelope schema `stem4humanity.trace-event/v1` (stable artifact format id)
  - `algofinder/README.md:9`, root `README.md` — umbrella "stem4humanity" prose
  - `algofinder/config/config.toml:2` — header comment "stem4humanity / algofinder"
  - `algofinder/public/results/benchmark.json` — generated output (fs paths)
- Stale compiled artifacts (gitignored, harmless, MUST NOT be deleted — no-delete rule):
  `.venv/lib/python3.13/site-packages/scipy/**/__pycache__/*.pyc`,
  `algofinder/algofinder/util/__pycache__/tracing.cpython-313.pyc`,
  any leftover `stem4humanity.egg-info/` (new `algofinder.egg-info/` is active).

## Workspace rules (binding)

- Only touch the repository worktree (project under `projects/stem4humanity/`).
- NEVER delete files/dirs/symlinks; renames and content edits only.
- No test files; verify by running the real pipeline (`generate`/`train`/`benchmark`/`report`).
- Git: repo root is `projects/stem4humanity`, branch `refactor/rename-package-algofinder`
  (off `main`). Index has the rename staged (`RM` entries); content edits are unstaged.
  Do NOT commit, merge, or push unless the user explicitly asks.

## Git state snapshot (at handoff)

- Branch: `refactor/rename-package-algofinder` (no commits on it yet)
- Staged: directory rename + package name updates via `git mv`/edits
- Unstaged: README/config/docs/scratch-script/benchmark edits, regenerated
  `public/results/benchmark.json` + `leaderboard.md` (pipeline ran fine)
