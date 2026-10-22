# Periodic Holstein HEOM-TT example

This example is intentionally procedural: `config.jl` defines the default
controls, `model.jl` builds the Hamiltonian and HEOM problem, `utils.jl` holds
the propagation/measurement helpers, and the two short entry scripts run the
calculation.

From the repository root, install the example environment and run:

```bash
julia --project=examples/holstein examples/holstein/equilibrate.jl
julia --project=examples/holstein examples/holstein/current_correlation.jl
```

The first command writes an equilibrated state, minimal metadata, CSV
diagnostics, a population/coherence plot, and an equilibration-rank plot to
`examples/holstein/output/`.
The second command reloads that state, checks its layout against the current
configuration, and writes the current correlation and rank plots.

The default initial state is localized on `initial_site`. Use a merged
`NamedTuple` to select the bare electronic Gibbs state:

```julia
include("examples/holstein/equilibrate.jl")
config = merge(DEFAULT_CONFIG, (initial_condition=:gibbs,))
equilibrate_main(config)
```

The solver tolerances are independent: use
`equilibration_tamen_tolerance` and `correlation_tamen_tolerance` to tune the
two calculations separately.

Set `blas_threads` in the configuration to control the process-wide BLAS
thread count. The default is `1`; this avoids oversubscription with Julia/TT
work. `BLAS.set_num_threads` is applied at the beginning of each entry point.

Set `decomposition_method` to `:tpsd` (default) or `:psd` to select QFiND's
truncated or full Padé spectral decomposition. `tpsd_tolerance` is used only
for `:tpsd`.

tAMEn propagation prints step, time, trace error/correlation, and TT-rank
diagnostics at step zero, the final step, and every `config.progress_interval`
steps.

All HEOM auxiliary modes start in their vacuum component. Each Crank–Nicolson
tAMEn step preserves the supplied HEOM trace observable within the requested
solver tolerance (`correct_2_norm=false`); no post-step normalization is
performed. Equilibration reports every site population, every unique
site-pair coherence, the trace, and maximum/mean TT rank.

The current source is `J ρ_eq`, and the reported unsymmetrized correlation is
`Tr[J exp(L t)(J ρ_eq)]`. The metadata intentionally contains only the state
identity, initial-condition choice, equilibration duration, hierarchy sizes,
and TT dimensions. If those physical inputs change, rerun equilibration before
running the correlation script.
