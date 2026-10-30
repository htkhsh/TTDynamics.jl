# Electron–lattice correlation implementation plan

**Goal:** Measure the user's equal-time connected correlation in both periodic examples:
`C_nB(r,t) = sum_j (<n_j B_mod(j+r,N)> - <n_j><B_mod(j+r,N)>)`.

**Architecture:** A shared TTDynamics measurement contracts only the root and
first-tier scaled ADOs. Both examples sample it during physical equilibration,
then export a distance-by-time CSV and heatmap. The public measurement also
accepts loaded checkpoints. Current-source propagation is not a density state
and is not used for these equal-time expectations.

**Tech stack:** Julia, TTSolver TT contractions, HEOMKit NoiseExp, CairoMakie.

## Constraints and decisions

- Preserve the user's existing configuration, scripts, and generated data.
- Work in the existing checkout so the uncommitted example changes are included.
- Bath index corresponds to the lattice site in both models; Fröhlich electronic
  weights are already in the coupling operators and are not applied again.
- First-tier weight is `sqrt(noise.abs_coeff[k])`, using expanded bath mode ranges.
- Normalize moments by the root trace; reject zero/nonfinite trace and missing
  first-tier components rather than silently reporting zero.
- Keep complex measurements; coordinates and correlations have units `fs^-1`.
- No dense hierarchy expansion in production and no new package dependencies.

## Tasks

- [x] Add synthetic multi-mode, asymmetric periodic correlation tests, including
      disconnected cancellation, trace rescaling, higher-tier exclusion, and
      invalid state/hierarchy checks. Run and observe the missing API failure.
- [x] Implement/export `electron_lattice_correlation(state, system)` in
      `src/heom_observables.jl`; reuse vacuum prefix/suffix contractions and
      return distances, populations, bath means, joint moments, raw and connected
      correlations. Run focused tests.
- [x] Test and integrate measurement into both examples' equilibration helpers,
      CSV exporters, heatmap plotters, and entry points. Document checkpoint
      measurement and generated filenames in both READMEs.
- [x] Run focused dense-reference propagation and output smoke checks in both
      models, run the repository suite, and review the diff. Report any baseline
      test failures separately from feature verification.

## Review focus

Unequal numbers of bath terms, asymmetric positive distances, complex ADOs,
root trace drift, hierarchy size one, factorized states, and pre-existing
configuration-dependent test failures.

## Verification notes

- Core tests: 18 passed, after first observing failure for the missing API.
- Integration first observed failures for the missing measurement field in both
  examples. The implemented integration passed CSV, heatmap, and dense-reference
  checks; a test-only amplitude assertion used too short a localized trajectory.
  Use factorized bare electronic Gibbs initial states to resolve a larger signal.
- Existing-suite baseline stops at `test/peierls_current.jl` because the user's
  working tree has already deleted `examples/peierls/validate_dense.jl`.
- Independent review found no critical or important issues.
- Final focused run: 18 core tests and 32 example integration checks passed,
  including dense propagation comparisons, CSV values, and heatmaps.
- Both normal equilibration entry points passed 8 further smoke checks with
  Brownian PSD baths, including checkpoint reload and matching measurements.
- Final repository-suite run passed all tests reached (including the 50 new
  checks) and then stopped at the same pre-existing missing Peierls file.
  `git diff --check` passed.
- Julia verification uses `--compiled-modules=existing` to avoid writing caches
  outside the writable workspace; no dependency changes are needed.
