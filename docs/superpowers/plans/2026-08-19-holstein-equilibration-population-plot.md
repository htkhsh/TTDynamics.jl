# Holstein Equilibration Population Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically save a site-population PNG whenever the Holstein current-correlation equilibration executable publishes its checkpoint outputs.

**Architecture:** Keep CairoMakie presentation code in `plotting.jl` and consume the in-memory equilibration result. `equilibrate.jl` lazily includes that file only for its default plotter and uses latest-world binding lookup and invocation so package tests with injected plotters do not require CairoMakie. Generalize the existing atomic PNG publisher in the example utilities so both equilibration and current-correlation plots share the same no-clobber behavior, then treat the new PNG as part of the equilibration output set.

**Tech Stack:** Julia 1.12, CairoMakie, Julia `Test`, TTDynamics HEOM-TT example utilities

## Global Constraints

- Output filename is exactly `output/holstein_equilibration_populations.png`.
- Plot one labeled line per population row, with `Time (fs)` and `Population` axis labels.
- Use `result.times` and `result.populations` directly; do not reread the CSV.
- Apply the same `overwrite` setting to CSV, TT binary, TOML metadata, and PNG.
- A failed non-overwrite publication removes every artifact newly created by that invocation.
- Do not change HEOM equations, Brownian TPSD decomposition, physical parameters, output data formats, or simulation durations.
- Preserve pre-existing user changes under `examples/holstein`.

---

### Task 1: Publish the equilibration population plot

**Files:**
- Modify: `examples/holstein_current_correlation/utils.jl:719-750`
- Modify: `examples/holstein_current_correlation/equilibrate.jl`
- Create: `examples/holstein_current_correlation/plotting.jl`
- Modify: `examples/holstein_current_correlation/current_correlation.jl:85-95`
- Modify: `examples/holstein_current_correlation/README.md:45-66`
- Modify: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: `result.times`, `result.populations`, `_equilibrium_atomic_rename`, and CairoMakie `Figure`, `Axis`, `lines!`, `axislegend`, and `save`.
- Produces: `_save_equilibration_population_plot(path, result) -> String` from `plotting.jl`, `_default_equilibration_plotter(path, result) -> String` with lazy latest-world resolution, `write_plot_png(path, result, plotter; overwrite=false) -> String`, and an added `png_path` field in `_equilibration_output_paths` and `equilibrate_main` results.

- [ ] **Step 1: Write failing path and orchestration tests**

Extend the equilibrium path test to require:

```julia
@test basename(paths.png_path) == "holstein_equilibration_populations.png"
```

Extend the synthetic `save_equilibration_outputs` test with an injected plotter that writes a small PNG fixture and records its input:

```julia
plot_calls = Any[]
plotter = (path, result) -> begin
    push!(plot_calls, (path, result))
    write(path, "synthetic png")
    path
end
outputs = save_equilibration_outputs(
    config,
    decomposition,
    problem,
    result;
    output_directory,
    plotter,
)
@test isfile(outputs.png_path)
@test only(plot_calls)[2] === result
```

Also assert that a pre-existing PNG is rejected when `overwrite=false`, and that an injected plotting failure removes the newly created CSV, TT binary, metadata, and PNG paths.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
julia --startup-file=no --project=. test/holstein_current_correlation.jl
```

Expected: failure because `_equilibration_output_paths` has no `png_path` and `save_equilibration_outputs` has no `plotter` keyword.

- [ ] **Step 3: Generalize the atomic PNG publisher**

Rename `write_current_correlation_png` to a presentation-neutral helper while preserving its implementation:

```julia
function write_plot_png(path::AbstractString, result, plotter;
                        overwrite::Bool=false)::String
    target = String(path)
    !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
    temporary_path = nothing
    try
        temporary_path, io = mktemp(dirname(abspath(target)))
        close(io)
        png_temporary_path = "$(temporary_path).png"
        mv(temporary_path, png_temporary_path)
        temporary_path = png_temporary_path
        plotter(temporary_path, result)
        isfile(temporary_path) || throw(ArgumentError("plotter did not write a PNG file"))
        !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
        if overwrite
            _equilibrium_atomic_rename(temporary_path, target)
            temporary_path = nothing
        else
            Base.Filesystem.hardlink(temporary_path, target)
        end
        return target
    finally
        temporary_path === nothing || rm(temporary_path; force=true)
    end
end
```

Update `current_correlation_main` to call:

```julia
png_path = write_plot_png(paths.png_path, result, plotter; overwrite)
```

Update existing tests to call `write_plot_png`; do not retain a redundant compatibility alias because this is example-internal API.

- [ ] **Step 4: Add the lazily loaded CairoMakie population renderer**

Create `plotting.jl` with `using CairoMakie` and define:

```julia
function _save_equilibration_population_plot(path::AbstractString, result)::String
    length(result.times) == size(result.populations, 2) ||
        throw(ArgumentError("population column count must equal times length"))
    figure = Figure(size=(900, 600))
    axis = Axis(
        figure[1, 1];
        xlabel="Time (fs)",
        ylabel="Population",
        title="Holstein HEOM-TT equilibration populations",
    )
    for site in axes(result.populations, 1)
        lines!(axis, result.times, result.populations[site, :];
               linewidth=2, label="Site $site")
    end
    axislegend(axis; position=:rt)
    save(String(path), figure)
    return String(path)
end
```

- [ ] **Step 5: Add PNG publication to the equilibrium transaction**

Add:

```julia
const DEFAULT_EQUILIBRATION_PNG_PATH = joinpath(
    DEFAULT_EQUILIBRATION_OUTPUT_DIRECTORY,
    "holstein_equilibration_populations.png",
)
```

Return `png_path` from `_equilibration_output_paths`. In `equilibrate.jl`, add
`_default_equilibration_plotter` that `@__DIR__`-includes `plotting.jl` on
demand and resolves/invokes `_save_equilibration_population_plot` with
`Base.invokelatest`. Add `plotter=_default_equilibration_plotter` to
`save_equilibration_outputs` and `equilibrate_main`, forwarding it through both
functions. Publish the plot
with:

```julia
png_path = write_plot_png(paths.png_path, result, plotter; overwrite)
```

Track every path newly published during a non-overwrite invocation. On any
exception, remove those new paths in reverse publication order before
rethrowing. Return and print `png_path` with the other outputs.

- [ ] **Step 6: Verify focused GREEN and real rendering**

Run:

```bash
julia --startup-file=no --project=. test/holstein_current_correlation.jl
```

Expected: all focused assertions pass.

Then run a small rendering-only check that does not propagate HEOM:

```bash
julia --startup-file=no --project=examples/holstein_current_correlation -e 'include("examples/holstein_current_correlation/equilibrate.jl"); mktempdir() do d; result=(times=[0.0,1.0], populations=[1.0 0.8; 0.0 0.2]); p=_save_equilibration_population_plot(joinpath(d,"populations.png"), result); @assert isfile(p); end'
```

Expected: exit status 0 and a valid temporary PNG is created.

- [ ] **Step 7: Document the automatic plot output**

Add the following item to the README equilibration output list:

```markdown
- `output/holstein_equilibration_populations.png`
```

State immediately after the list that the PNG plots each site's population
against time and is generated automatically by `equilibrate.jl`.

- [ ] **Step 8: Run package verification and commit**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
git diff --check
```

Expected: all tests pass; existing intentionally guarded imports may remain
reported as `Broken`. Verify the pre-existing user modifications under
`examples/holstein` are untouched, then commit only the five task files:

```bash
git add examples/holstein_current_correlation/utils.jl \
        examples/holstein_current_correlation/equilibrate.jl \
        examples/holstein_current_correlation/plotting.jl \
        examples/holstein_current_correlation/current_correlation.jl \
        examples/holstein_current_correlation/README.md \
        test/holstein_current_correlation.jl
git commit -m "feat: plot Holstein equilibration populations"
```
