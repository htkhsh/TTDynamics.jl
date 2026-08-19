# Holstein Equilibration Population Plot

## Goal

Generate a site-population plot automatically whenever
`examples/holstein_current_correlation/equilibrate.jl` completes an
equilibration run.

## Output

Save the figure as
`output/holstein_equilibration_populations.png` beside the existing CSV,
TT binary, and TOML metadata. The horizontal axis is `Time (fs)`, the vertical
axis is `Population`, and one labeled line (`Site 1`, `Site 2`, ...) is drawn
for each row of `result.populations`.

## Architecture

`plotting.jl` imports CairoMakie and owns the small population plotting
function because plotting is executable presentation logic, not HEOM-TT model
logic. `equilibrate.jl` remains free of a top-level plotting dependency: its
default plotter includes `plotting.jl` from `@__DIR__` only when a real plot is
requested, then resolves and invokes the renderer through latest-world binding
lookup. The plot consumes `result.times` and `result.populations` directly; it
does not read the CSV back from disk. Injected plotters remain direct and never
trigger the lazy renderer.

Add the PNG path to `_equilibration_output_paths`. Publish the figure through
the existing atomic/no-clobber PNG helper used by the current-correlation
workflow, so `overwrite=false` cannot replace an existing plot during a race.
The plotter remains injectable in `save_equilibration_outputs` and
`equilibrate_main`, allowing tests to avoid a graphical backend when they test
orchestration rather than rendering.

## Publication and Failure Behavior

All four artifacts—the diagnostics CSV, TT binary, TOML metadata, and PNG—use
the same `overwrite` setting. Before a non-overwrite run, reject the operation
if any target already exists. If publication fails after creating artifacts in
a non-overwrite run, remove every artifact newly created by that invocation so
that a partial equilibrium checkpoint is not presented as complete.

For `overwrite=true`, retain the existing documented limitation: a previously
replaced artifact cannot be restored automatically after a later publication
failure. Production replacement runs should use a fresh output directory when
rollback is required.

## Testing

Use a small synthetic result to verify the PNG path and automatic invocation
from the equilibrium save workflow. Add regressions for lazy renderer loading,
no-overwrite protection, atomic PNG publication, and cleanup after an injected
plotting failure. Run the focused Holstein current-correlation test file and
the complete package test suite; separately run a small real-rendering check
in the example environment.

## Documentation

Update the example README output list to include
`output/holstein_equilibration_populations.png` and state that it is generated
by `equilibrate.jl`.
