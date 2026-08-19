# Holstein Current-Correlation Rank Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Print useful current-correlation progress and TT ranks and automatically save a separate TT-rank history PNG.

**Architecture:** Add an optional progress callback to the current-correlation runner while preserving the fixed-step propagator and numerical result. Extend the executable's atomic output transaction with a second plot generated from the rank arrays already recorded in the result.

**Tech Stack:** Julia, TTDynamics/TTSolver TT types, CairoMakie, Julia `Test`.

## Global Constraints

- Report step 0, every `config.progress_interval` steps, and the final step.
- Progress lines contain step/total, time in fs, maximum rank, mean rank, and elapsed wall-clock time.
- Print complete equilibrium and final source TT-rank vectors.
- Save `holstein_current_correlation_ranks.png` separately; keep the existing correlation PNG and CSV schema unchanged.
- Preserve 1000 fs equilibration, 200 fs correlation, QFiND/TPSD, HEOM-TT dynamics, solver parameters, and correlation definition.
- Preserve atomic no-clobber and rollback behavior for every published output.

---

### Task 1: Progress and Rank Diagnostics

**Files:**
- Modify: `examples/holstein_current_correlation/utils.jl`
- Modify: `examples/holstein_current_correlation/current_correlation.jl`
- Test: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: `tt_ranks(::TTTensor)`, `HolsteinConfig.progress_interval`, and the existing per-step rank observations.
- Produces: `run_current_correlation(...; progress_callback)` where the callback receives one named tuple containing `step`, `step_count`, `time_fs`, `maximum_rank`, `mean_rank`, and `elapsed_seconds`.

- [ ] **Step 1: Write failing progress tests**

Add a zero-Liouvillian fixture with five steps and `progress_interval=2`. Capture
callback records and assert reported steps are `[0, 2, 4, 5]`, times match the
configured time step, ranks match the result arrays at those indices, elapsed
values are finite/nonnegative, and the numerical result equals a run with no
callback. Add a main-level injected runner test that checks the loaded
equilibrium rank vector and final state rank vector appear in captured stdout.

- [ ] **Step 2: Run the focused test to verify RED**

Run:
`julia --project=. test/holstein_current_correlation.jl`

Expected: FAIL because `progress_callback` and the loaded-rank message do not
exist.

- [ ] **Step 3: Implement minimal progress reporting**

Add `progress_callback=nothing` to `run_current_correlation`. Measure elapsed
time with `time_ns()`, create each existing observation once, and invoke the
callback only for step 0, interval boundaries, and final step. Add a default
formatter in `current_correlation.jl`, pass it from `current_correlation_main`,
and print `tt_ranks(equilibrium_state)` after metadata validation plus
`tt_ranks(result.state)` at completion. Avoid printing from library code when
the callback is `nothing`.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run:
`julia --project=. test/holstein_current_correlation.jl`

Expected: all current-correlation tests pass.

- [ ] **Step 5: Commit the diagnostic change**

```bash
git add examples/holstein_current_correlation/utils.jl examples/holstein_current_correlation/current_correlation.jl test/holstein_current_correlation.jl
git commit -m "feat: report Holstein current TT ranks"
```

### Task 2: Automatic Rank Plot

**Files:**
- Modify: `examples/holstein_current_correlation/current_correlation.jl`
- Modify: `examples/holstein_current_correlation/README.md`
- Test: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: `result.times`, `result.maximum_rank`, `result.mean_rank`, and `write_plot_png`.
- Produces: `_save_current_correlation_rank_plot(path, result)` and returned `rank_png_path` from `current_correlation_main`.

- [ ] **Step 1: Write failing rank-output tests**

Extend output-path tests to require
`holstein_current_correlation_ranks.png`. Inject distinct correlation/rank
plotters, assert each receives the result and publishes its target, assert all
three public outputs are preflighted with `overwrite=false`, and inject a
rank-plot failure after CSV/correlation publication to assert invocation-owned
files are rolled back.

- [ ] **Step 2: Run the focused test to verify RED**

Run:
`julia --project=. test/holstein_current_correlation.jl`

Expected: FAIL because `rank_png_path` and the rank plotter do not exist.

- [ ] **Step 3: Implement the rank plot and transaction**

Add `DEFAULT_CURRENT_CORRELATION_RANK_PNG_PATH`, include it in output paths and
no-clobber preflight, and render maximum/mean rank lines with labels `Time (fs)`
and `TT rank`. Publish through `write_plot_png`. Track only paths successfully
published by this invocation and remove them in reverse order after a later
failure when `overwrite=false`. Return and print `rank_png_path`.

- [ ] **Step 4: Update README**

Document the progress cadence, full equilibrium/final rank lines, CSV rank
columns, and the new `output/holstein_current_correlation_ranks.png` file.

- [ ] **Step 5: Run focused and full verification**

Run:
`julia --project=. test/holstein_current_correlation.jl`

Run:
`julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: both commands exit 0 with all tests passing.

- [ ] **Step 6: Commit the plot change**

```bash
git add examples/holstein_current_correlation/current_correlation.jl examples/holstein_current_correlation/README.md test/holstein_current_correlation.jl
git commit -m "feat: plot Holstein current TT ranks"
```
