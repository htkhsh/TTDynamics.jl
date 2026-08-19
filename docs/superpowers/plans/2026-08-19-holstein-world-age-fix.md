# Holstein Builder World-Age Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the lazily loaded Holstein decomposition and HEOM problem builders callable on Julia 1.12 without world-age warnings or errors.

**Architecture:** Preserve the current lazy include boundary in `equilibrate.jl`. Resolve each newly loaded builder through the current module in the latest world, then cross the call boundary with `Base.invokelatest`, while leaving explicitly injected builders on the normal direct-call path.

**Tech Stack:** Julia 1.12, `Base.invokelatest`, Julia `Test`, TTDynamics example environment

## Global Constraints

- Keep lazy loading of `examples/holstein/holstein_brownian_heomtt.jl`.
- Do not change the HEOM equations, Brownian TPSD decomposition, physical defaults, output formats, or simulation durations.
- Do not modify or discard pre-existing user changes under `examples/holstein`.

---

### Task 1: Make lazy Holstein builders world-age safe

**Files:**
- Modify: `examples/holstein_current_correlation/equilibrate.jl:13-30`
- Modify: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: `_load_holstein_builders()`, `HolsteinConfig`, `decompose_brownian_bcf(config)`, and `build_holstein_heomtt(config, decomposition)`.
- Produces: unchanged `_default_decomposition_builder(config)` and `_default_problem_builder(config, decomposition)` interfaces with Julia 1.12-safe lazy dispatch.

- [ ] **Step 1: Write the failing subprocess regression**

Add a test that copies `equilibrate.jl` into a temporary fixture with a minimal lazily included builder file. The fixture must define `decompose_brownian_bcf` and `build_holstein_heomtt` only during the first builder call, then invoke both default wrappers from a function compiled before the include. Run it in a fresh Julia process with `--depwarn=error` and assert a zero exit status:

```julia
command = `$(Base.julia_cmd()) --startup-file=no --depwarn=error $fixture_script`
process = run(ignorestatus(command))
@test success(process)
```

The fixture must exercise the production wrapper logic rather than merely testing `Base.invokelatest` in isolation.

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
julia --startup-file=no --project=. test/holstein_current_correlation.jl
```

Expected: the new subprocess test fails because direct access to the lazily defined global binding becomes a deprecation error or because the method is too new for the caller's world.

- [ ] **Step 3: Implement the minimal world-age-safe calls**

Anchor the lazy include to `equilibrate.jl` so it works independently of the process working directory. After `_load_holstein_builders()`, retrieve each function dynamically and invoke it in the latest world:

```julia
function _load_holstein_builders()
    if !isdefined(@__MODULE__, :decompose_brownian_bcf) ||
       !isdefined(@__MODULE__, :build_holstein_heomtt)
        include(joinpath(@__DIR__, "..", "holstein", "holstein_brownian_heomtt.jl"))
    end
    return nothing
end

function _default_decomposition_builder(config)
    _load_holstein_builders()
    builder = Base.invokelatest(getfield, @__MODULE__, :decompose_brownian_bcf)
    return Base.invokelatest(builder, config)
end

function _default_problem_builder(config, decomposition)
    _load_holstein_builders()
    builder = Base.invokelatest(getfield, @__MODULE__, :build_holstein_heomtt)
    return Base.invokelatest(builder, config, decomposition)
end
```

- [ ] **Step 4: Verify the focused regression is GREEN**

Run:

```bash
julia --startup-file=no --project=. test/holstein_current_correlation.jl
```

Expected: all focused tests pass, including the fresh-process world-age regression, with no world-age warning under `--depwarn=error`.

- [ ] **Step 5: Verify the actual example load path**

Run from the repository root using the example environment:

```bash
julia --startup-file=no --depwarn=error --project=examples/holstein_current_correlation -e 'include("examples/holstein_current_correlation/equilibrate.jl"); decomposition = _default_decomposition_builder(HolsteinConfig(site_count=2)); @assert decomposition !== nothing'
```

Expected: exit status 0 with no world-age warning or error. This deliberately stops before the 1000 fs equilibration.

- [ ] **Step 6: Run the package test suite**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all TTDynamics tests pass; existing intentionally broken guarded-import tests may remain reported as `Broken`.

- [ ] **Step 7: Review and commit**

Run `git diff --check`, verify that the five pre-existing modified Holstein result/utility files remain untouched by this task, then commit only the regression and implementation:

```bash
git add examples/holstein_current_correlation/equilibrate.jl test/holstein_current_correlation.jl
git commit -m "fix: support Julia 1.12 lazy Holstein builders"
```
