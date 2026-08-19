# Holstein Current-Correlation Rank Progress Design

## Goal

Make the 200 fs current-correlation calculation visibly progress, report the
TT-rank growth that controls its cost, and automatically save a rank-history
plot without changing the HEOM-TT dynamics or current-correlation definition.

## Console diagnostics

`current_correlation_main` prints the complete TT-rank vector immediately
after loading and validating the equilibrium state. `run_current_correlation`
reports step 0, every `config.progress_interval` propagation steps, and the
final step (even when it is not an interval boundary). Each progress line
contains the current step and total steps, physical time in fs, maximum rank,
mean rank, and elapsed wall-clock time. The final propagated source rank vector
is printed after propagation. This keeps routine output compact while exposing
the two states whose complete bond structure is most useful for diagnosis.

Progress reporting is injectable and optional so numerical tests can capture
messages without depending on stdout. It must not alter the propagated state,
observation order, correlation values, or rank measurements.

## Rank plot and output transaction

The existing correlation plot remains unchanged. A second PNG,
`holstein_current_correlation_ranks.png`, plots maximum and mean TT rank against
time in fs using the arrays already stored in the result and CSV. The executable
generates this plot automatically after the CSV and correlation plot.

The rank PNG participates in the same no-clobber preflight and rollback rules
as the existing outputs. With `overwrite=false`, any existing CSV, correlation
PNG, or rank PNG aborts before the expensive decomposition and propagation. If
a later publication fails, files newly published by this invocation are
removed, while pre-existing or concurrently created files are not removed.

## Compatibility and scope

- Keep the default equilibrium time at 1000 fs and correlation time at 200 fs.
- Keep QFiND/TPSD, Brownian parameters, HEOM construction, CN-tAMEn settings,
  current operator, and unsymmetrized correlation definition unchanged.
- Keep the current-correlation CSV schema unchanged; it already contains
  `max_rank` and `mean_rank`.
- Add no dependencies; CairoMakie is already an example dependency.
- Update the example README with the new console output and rank PNG.

## Verification

Tests cover interval/final progress selection, message data, full loaded/final
rank output, separate rank-plot publication, overwrite preflight, and rollback
when either plot writer fails. Existing zero-time correlation and CSV tests
continue to protect the numerical definition and output schema.
