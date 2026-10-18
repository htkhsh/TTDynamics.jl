# Holstein Example Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the framework-like Holstein example with a short, readable scientific workflow while preserving its HEOM physics, trace-conserving tAMEn propagation, diagnostics, and two-stage checkpoint/correlation flow.

**Architecture:** Keep the six existing example files, but replace the custom configuration and persistence framework with a named-tuple configuration, small computational helpers, direct output writes, and two linear entry-point workflows. Checkpoints retain a TT binary plus a minimal TOML sidecar whose extra keys are ignored for compatibility.

**Tech Stack:** Julia 1.11, TTDynamics, TTSolver, HEOMKit, QFiND, TOML, CairoMakie, Test

**Spec:** `docs/superpowers/specs/2026-10-16-holstein-example-simplification-design.md`

## Global Constraints

- Change only `examples/holstein`, its tests, README, and this plan; do not modify lattice Fröhlich.
- Keep `config.jl`, `model.jl`, `utils.jl`, `equilibrate.jl`, `current_correlation.jl`, and `plotting.jl` as separate files.
- Keep localized and bare-system Gibbs initial conditions; all HEOM auxiliary modes start in vacuum.
- Every propagated density/source step uses the trace observable in `conserved_observables`, sets `correct_2_norm=false`, and reuses one `TAMEnWorkspace` per run.
- Never normalize or rescale a propagated state after tAMEn.
- Preserve the current output filenames, CSV physical units, and population/coherence/current/rank figures.
- Entry scripts remain safe to `include`; computation starts only under the program-file guard.
- Direct writes replace existing example outputs; atomic replacement, rollback, no-clobber policy, and dependency-injection hooks are intentionally removed.
- Use line count only as a readability guard: target at most 400 lines for `utils.jl` and 120 lines for each entry script, without compressing distinct operations onto dense one-liners.

## Review Focus

- A duration that is not an integer multiple of `time_step_fs` must fail before propagation; test both equilibration and correlation durations in Task 2.
- An unsupported `initial_condition` must fail at initial-state construction; test `:invalid` in Task 2.
- A loaded checkpoint with missing required metadata or incompatible TT dimensions must fail, while an old sidecar with extra keys remains accepted; test all three cases in Task 3.
- The current source `J * rho` is not normalized, but its initial trace must be preserved by tAMEn; test this with a zero Liouvillian in Task 2.
- Direct output replacement may leave earlier files if a later write fails; document this behavior and test only successful replacement, not transactional recovery, in Task 3.

---

### Task 1: Simplify configuration and model construction

**Files:**
- Modify: `examples/holstein/config.jl`
- Modify: `examples/holstein/model.jl`
- Modify: `test/holstein_current_correlation.jl`

**Interfaces:**
- Produces: `DEFAULT_CONFIG::NamedTuple`, `EQUILIBRATION_TIME_FS::Float64`, and `CORRELATION_TIME_FS::Float64`.
- Produces: `holstein_hamiltonian(config)::Matrix{ComplexF64}`.
- Produces: `site_projectors(site_count)::Vector{Matrix{ComplexF64}}`.
- Produces: `decompose_bath(config)::NamedTuple{(:exponents, :coefficients)}`.
- Produces: `build_problem(config, decomposition)::NamedTuple` with `system`, `liouvillian`, `trace_observable`, and `population_observables`.
- Produces: `current_operator(config)::Matrix{ComplexF64}`.

- [ ] **Step 1: Replace framework-oriented configuration tests with failing named-tuple contract tests**

In `test/holstein_current_correlation.jl`, add assertions that `DEFAULT_CONFIG` has the approved physical/numerical keys, excludes `final_time_fs` and the three removed BCF-validation controls, supports `merge(DEFAULT_CONFIG, (; initial_condition=:gibbs))`, and that `EQUILIBRATION_TIME_FS == 1000.0` and `CORRELATION_TIME_FS == 200.0`.

- [ ] **Step 2: Run the focused configuration test and verify RED**

Run a Julia `-e` test that includes `config.jl` and asserts the `DEFAULT_CONFIG` contract.

Expected: FAIL because `DEFAULT_CONFIG` is not defined and the custom `HolsteinCurrentCorrelationConfig` still exists.

- [ ] **Step 3: Replace the custom configuration type with constants**

Define the three constants from the interface block. Put all currently retained fields into `DEFAULT_CONFIG` with the existing defaults. Delete the constructor and general validation function.

- [ ] **Step 4: Add failing model tests for the compact API**

Test the literal two- and three-site Hamiltonians, site projectors, Hermitian current operator, a two-term synthetic decomposition passed to `build_problem`, and the real `decompose_bath` return property names `(:exponents, :coefficients)`.

- [ ] **Step 5: Run the focused model tests and verify RED**

Expected: FAIL because the compact model function names do not exist.

- [ ] **Step 6: Rewrite `model.jl` around the compact API**

Keep the existing physical formulas and QFiND TPSD checks, but remove references to the deleted configuration type and move the current-operator construction out of `utils.jl`. Do not add wrapper aliases for the deleted names.

- [ ] **Step 7: Run the focused configuration/model tests and verify GREEN**

Expected: all new configuration and model tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add examples/holstein/config.jl examples/holstein/model.jl test/holstein_current_correlation.jl
git commit -m "refactor: simplify Holstein configuration and model"
```

### Task 2: Replace `utils.jl` with the scientific propagation core

**Files:**
- Rewrite: `examples/holstein/utils.jl`
- Modify: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: `DEFAULT_CONFIG` fields and the HEOM problem returned by `build_problem`.
- Produces: `step_count(duration_fs, time_step_fs; allow_zero=false)::Int`.
- Produces: `gibbs_density(system, temperature_K)::Matrix{ComplexF64}`.
- Produces: `initial_heom_state(config, system)::TTTensor`.
- Produces: `current_heom_operators(config, problem)::NamedTuple` with `left_action` and `observable`.
- Produces: `tamen_step(state, liouvillian, config, trace_observable, workspace)::TTTensor`.
- Produces: `measure_density(state, problem)::NamedTuple` with populations, coherence pairs, complex coherences, trace, maximum rank, and mean rank.
- Produces: `run_equilibration(config, problem; duration_fs)::NamedTuple`.
- Produces: `run_current_correlation(equilibrium_state, config, problem; duration_fs)::NamedTuple`.

- [ ] **Step 1: Write failing initial-state and time-grid tests**

Use a small bath-free `HEOMTTSystem` to assert the localized density, the hand-computed Gibbs density, vacuum auxiliary cores, rejection of `:invalid`, integral step counts, zero correlation duration, and rejection of non-integral/negative durations.

- [ ] **Step 2: Run the focused tests and verify RED**

Expected: FAIL because the new utility names do not exist.

- [ ] **Step 3: Implement time-grid and initial-state helpers**

Use the shifted Hermitian eigendecomposition already proven in the current implementation. Construct explicit ket, bra, and vacuum TT cores without a final rounding step that could perturb Hermiticity or trace.

- [ ] **Step 4: Write failing tAMEn and measurement tests**

With a zero Liouvillian, assert that one step preserves the exact snapshot within solver tolerance, preserves the supplied trace, returns all `i < j` coherence pairs, and reports TT ranks. Add a source-state case whose nonunit initial trace remains unchanged.

- [ ] **Step 5: Run the focused tests and verify RED**

Expected: FAIL because the compact propagation functions are missing.

- [ ] **Step 6: Implement compact propagation and measurement**

Call `tt_kron`, keyword-only `tamen`, and `extract_snapshot`. Pass `conserved_observables=[trace_observable]`, `correct_2_norm=false`, and the shared workspace. Return the snapshot directly without `tt_round`, normalization, or rescaling. Keep one straightforward loop for equilibration and one for the current source, with progress printed at `progress_interval`.

- [ ] **Step 7: Run the utility tests and verify GREEN**

Expected: initial-state, time-grid, one-step propagation, trace, population, coherence, current-source, and rank tests pass.

- [ ] **Step 8: Confirm removed framework helpers are gone**

Search `examples/holstein/utils.jl` for `normalize_heom_state`, metadata schema field lists, atomic rename, staged writes, `overwrite`, and injected writer/plotter callbacks. Expected: no matches.

- [ ] **Step 9: Commit Task 2**

```bash
git add examples/holstein/utils.jl test/holstein_current_correlation.jl
git commit -m "refactor: reduce Holstein propagation helpers"
```

### Task 3: Linearize plotting, checkpoint, equilibration, and correlation workflows

**Files:**
- Modify: `examples/holstein/utils.jl`
- Rewrite: `examples/holstein/plotting.jl`
- Rewrite: `examples/holstein/equilibrate.jl`
- Rewrite: `examples/holstein/current_correlation.jl`
- Modify: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: Task 1 model builders and Task 2 propagation functions.
- Produces: `checkpoint_metadata(config, problem, state, equilibration_time_fs)::Dict{String,Any}` with exactly the seven required keys from the spec.
- Produces: `write_metadata(path, metadata)::String`, `read_metadata(path)::Dict{String,Any}`, and `check_checkpoint(state, metadata, config, problem)::TTTensor`.
- Produces: `write_equilibration_csv(path, result)::String` and `write_current_csv(path, result)::String`.
- Produces: `save_equilibration_plot(path, result)::String`, `save_current_plot(path, result)::String`, and `save_rank_plot(path, result)::String`.
- Produces: `equilibrate_main(config=DEFAULT_CONFIG; equilibration_time_fs=EQUILIBRATION_TIME_FS, output_directory=joinpath(@__DIR__, "output"))::NamedTuple`.
- Produces: `current_correlation_main(config=DEFAULT_CONFIG; correlation_time_fs=CORRELATION_TIME_FS, output_directory=joinpath(@__DIR__, "output"))::NamedTuple`.

- [ ] **Step 1: Write failing plot contract tests**

Using synthetic result arrays, assert that the equilibration figure has population and all-pair coherence panels, the current figure has real/imaginary curves, and the rank figure has maximum/mean curves. Test the three direct save functions against temporary PNG paths.

- [ ] **Step 2: Run the plot tests and verify RED**

Expected: FAIL because the simplified plot function names do not exist.

- [ ] **Step 3: Simplify `plotting.jl`**

Keep the existing visual content, delete underscored lazy-loader-oriented names and redundant input validation, and implement the three direct save functions.

- [ ] **Step 4: Write failing minimal-checkpoint tests**

Assert the exact seven metadata keys and literal values for a small problem; `initial_condition` is stored as a string and numeric values keep TOML-native scalar/vector types. Write/read TOML directly, accept a copied metadata dictionary with an extra old key, reject a missing required key, reject wrong site/hierarchy dimensions, and reject a non-`TTTensor` checkpoint.

- [ ] **Step 5: Run the checkpoint tests and verify RED**

Expected: FAIL because the minimal persistence API is not implemented.

- [ ] **Step 6: Implement minimal metadata and CSV helpers**

Use direct `open(..., "w")`, `TOML.print`, `TOML.parsefile`, and comma-separated output. Writers replace existing paths. The equilibration CSV includes population columns, real/imaginary columns for every coherence pair, trace, maximum rank, and mean rank. The correlation CSV keeps its current units and rank columns.

- [ ] **Step 7: Write failing workflow tests**

For a two-site, one-step small configuration, run `equilibrate_main`, assert the four output files and finite trace diagnostics, then run `current_correlation_main` with zero correlation duration and assert the three correlation outputs. Include both scripts in a fresh module and assert no output appears until a main function is called.

- [ ] **Step 8: Run the workflow tests and verify RED**

Expected: FAIL because the entry scripts still expose the framework-style orchestration and injection hooks.

- [ ] **Step 9: Rewrite `equilibrate.jl` as a linear workflow**

Include `config.jl`, `model.jl`, `utils.jl`, and `plotting.jl` normally. In `equilibrate_main`, visibly perform decomposition, model construction, initial-state selection through `run_equilibration`, diagnostics, direct writes, and plotting in execution order. Use the existing output basenames and a program-file guard.

- [ ] **Step 10: Rewrite `current_correlation.jl` as a linear reload workflow**

In `current_correlation_main`, visibly rebuild the model, require/load/check the checkpoint, run the correlation, write CSV and both PNGs, and print ranks/paths. Remove builder, runner, plotter, overwrite, staged-output, and lazy-loader code. Keep the program-file guard.

- [ ] **Step 11: Run the plot, checkpoint, and workflow tests and verify GREEN**

Expected: metadata compatibility, direct replacement, small-system two-stage workflow, and safe-include tests all pass.

- [ ] **Step 12: Commit Task 3**

```bash
git add examples/holstein/utils.jl examples/holstein/plotting.jl examples/holstein/equilibrate.jl examples/holstein/current_correlation.jl test/holstein_current_correlation.jl
git commit -m "refactor: linearize Holstein example workflows"
```

### Task 4: Simplify documentation and the final test surface

**Files:**
- Modify: `examples/holstein/README.md`
- Modify: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: all interfaces and output formats from Tasks 1 through 3.
- Produces: a concise README and the final behavior-oriented test surface.

- [ ] **Step 1: Remove tests for deleted machinery and update README**

Delete testsets for atomic publication, rollback, no-clobber races, dependency injection, lazy loading, and world-age workarounds. Keep the scientific and end-to-end tests. Update README examples to use `DEFAULT_CONFIG` plus `merge`, document direct replacement and the seven-key checkpoint contract, and instruct users to rerun equilibration after changing model parameters.

- [ ] **Step 2: Run the complete Holstein example test**

Run: `julia --project=. test/holstein_current_correlation.jl`

Expected: all retained scientific, workflow, checkpoint, CSV, and real-plot tests pass with no warnings or errors.

- [ ] **Step 3: Run the package suite**

Run: `julia --project=. test/runtests.jl`

Expected: all package tests pass.

- [ ] **Step 4: Verify readability and stale API removal**

Run `wc -l examples/holstein/*.jl`, list top-level functions with `rg`, search for deleted framework terms, and run `git diff --check`. Confirm the execution flow is visible in the two entry scripts, `utils.jl` contains only the approved core/persistence helpers, no deleted API remains, and the diff is whitespace-clean.

- [ ] **Step 5: Commit Task 4**

```bash
git add examples/holstein/README.md test/holstein_current_correlation.jl
git commit -m "docs: streamline Holstein example usage"
```
