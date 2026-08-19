# Holstein Example Default Overwrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make direct Holstein equilibration and correlation runs overwrite existing outputs by default while retaining opt-in no-clobber behavior.

**Architecture:** Change only the two executable main-function keyword defaults. Keep all low-level writer and save-helper defaults conservative and validate the main-level behavior with injected fast tests.

**Tech Stack:** Julia, TTDynamics, Julia `Test`.

## Global Constraints

- `equilibrate_main` and `current_correlation_main` default to `overwrite=true`.
- Explicit `overwrite=false` retains existing collision errors and rollback.
- Low-level writer/save-helper defaults remain `overwrite=false`.
- No numerical, file-format, filename, duration, or plotting changes.

---

### Task 1: Default Main-Entry Overwrite

**Files:**
- Modify: `examples/holstein_current_correlation/equilibrate.jl`
- Modify: `examples/holstein_current_correlation/current_correlation.jl`
- Modify: `examples/holstein_current_correlation/README.md`
- Test: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: existing `overwrite` keyword forwarding in both main functions.
- Produces: main-entry default `overwrite=true`; writer APIs remain unchanged.

- [ ] **Step 1: Write failing tests**

Use injected zero-duration/short-duration runners and plotters to call each main
entry point twice in one temporary output directory without specifying
`overwrite`. Assert the second call succeeds and replaces sentinel output
content. Preserve explicit `overwrite=false` collision assertions.

- [ ] **Step 2: Verify RED**

Run `julia --project=. test/holstein_current_correlation.jl`.

Expected: the second main call fails with `target already exists`.

- [ ] **Step 3: Implement the minimal default changes**

Change only the `overwrite::Bool=false` defaults on `equilibrate_main` and
`current_correlation_main` to `overwrite::Bool=true`. Do not change defaults on
`save_equilibration_outputs`, `save_current_correlation_outputs`, CSV writers,
PNG writers, metadata writers, or TT binary I/O.

- [ ] **Step 4: Update README**

State that direct execution replaces existing example outputs and that callers
can pass `overwrite=false` to request no-clobber behavior.

- [ ] **Step 5: Verify and commit**

Run `julia --project=. test/holstein_current_correlation.jl` and
`julia --project=. -e 'using Pkg; Pkg.test()'`; both must exit 0. Commit the
two main files, README, tests, design, and plan.
