# KSL Spin-Boson Ohmic Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone thermofield KSL example for the existing Ohmic spin-boson model and compare its populations with a HEOMKit HEOM reference.

**Architecture:** Put reusable example-level TT rank and population operations in `examples/tfd/ksl/utils.jl`, covered by lightweight package tests. Build a separate executable `examples/tfd/ksl/sb_ohmic.jl` that reproduces the existing Ohmic/QFiND/ESPRIT/HEOM setup, constructs the same TFD Hamiltonian, pads the localized state to an admissible rank profile capped at 30, and advances it with symmetric KSL.

**Tech Stack:** Julia 1.11+, TTDynamics, TTSolver, QFiND, HEOMKit, ExpFit, CairoMakie, DelimitedFiles.

## Global Constraints

- Add `examples/tfd/ksl/sb_ohmic.jl` and `examples/tfd/ksl/utils.jl`.
- Do not modify `examples/tfd/sb-ohmic/sb_ohmic.jl` or any existing dirty file.
- Use the same Ohmic spin-boson physical parameters and HEOM reference settings as the approved design.
- Set the default maximum TT rank to `rmax = 30`.
- Compute internal ranks as `min(rmax, left Hilbert dimension, right Hilbert dimension)` with boundary ranks one.
- Zero-pad the localized product state without changing the represented state.
- Evolve `dψ/dt + iHψ = 0` using `tt_ksl(...; symmetric=true, rmax=rmax)`.
- Do not run the full 500 fs scientific simulation in the automated test suite.

---

### Task 1: Admissible rank, padding, and population utilities

**Files:**
- Create: `examples/tfd/ksl/utils.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `TTTensor`, `tt_dims`, `tt_ranks`, and `tt_full` from TTSolver.
- Produces:
  - `admissible_tt_ranks(dims::AbstractVector{<:Integer}, rmax::Integer=30)::Vector{Int}`
  - `pad_tt_ranks(x::TTTensor, ranks::AbstractVector{<:Integer})::TTTensor`
  - `first_site_populations(psi::TTTensor)::Vector{Float64}`

- [ ] **Step 1: Add failing utility tests**

At the end of `test/runtests.jl`, include the new utility file and add:

```julia
include("../examples/tfd/ksl/utils.jl")

@testset "KSL example utilities" begin
    @test admissible_tt_ranks([2, 3, 4]) == [1, 2, 4, 1]
    @test admissible_tt_ranks([2, 3, 4], 3) == [1, 2, 3, 1]
    @test_throws ArgumentError admissible_tt_ranks(Int[], 30)
    @test_throws ArgumentError admissible_tt_ranks([2, 0, 4], 30)
    @test_throws ArgumentError admissible_tt_ranks([2, 3, 4], 0)

    x0 = initial_tt_state([SpinSpace(0.5), BosonSpace(3), BosonSpace(4)];
                          state_index=1, site=1)
    target_ranks = admissible_tt_ranks(tt_dims(x0), 30)
    padded = pad_tt_ranks(x0, target_ranks)

    @test tt_ranks(padded) == target_ranks
    @test tt_full(padded) == tt_full(x0)
    @test_throws ArgumentError pad_tt_ranks(x0, [1, 2, 1])
    @test_throws ArgumentError pad_tt_ranks(x0, [2, 2, 4, 1])
    @test_throws ArgumentError pad_tt_ranks(padded, [1, 1, 1, 1])

    spin_superposition = product_tt_state([
        ComplexF64[1 / sqrt(2), 1im / sqrt(2)],
        ComplexF64[1, 0, 0],
    ])
    @test first_site_populations(spin_superposition) ≈ [0.5, 0.5]
end
```

These tests catch wrong boundary ranks, missing Hilbert-space bounds, state-changing padding, rank shrinking, and incorrect complex population contraction.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: `SystemError` because `examples/tfd/ksl/utils.jl` does not exist.

- [ ] **Step 3: Implement overflow-safe admissible ranks**

Create `examples/tfd/ksl/utils.jl` with:

```julia
using LinearAlgebra
using TTSolver

function _capped_product(values, cap::Int)
    result = 1
    for raw_value in values
        raw_value > 0 ||
            throw(ArgumentError("physical dimensions must be positive"))
        raw_value >= cap && return cap
        value = Int(raw_value)
        result > div(cap, value) && return cap
        result *= value
        result >= cap && return cap
    end
    return result
end

function admissible_tt_ranks(
    dims::AbstractVector{<:Integer},
    rmax::Integer=30,
)
    isempty(dims) && throw(ArgumentError("dims must not be empty"))
    rmax > 0 || throw(ArgumentError("rmax must be positive"))
    all(>(0), dims) || throw(ArgumentError("physical dimensions must be positive"))

    cap = Int(rmax)
    ranks = ones(Int, length(dims) + 1)
    for i in 1:(length(dims) - 1)
        left = _capped_product(@view(dims[1:i]), cap)
        right = _capped_product(@view(dims[(i + 1):end]), cap)
        ranks[i + 1] = min(cap, left, right)
    end
    return ranks
end
```

If review finds that the overflow guard can be simplified without weakening the contract, keep the public behavior and use an equivalent checked/capped multiplication.

- [ ] **Step 4: Implement state-preserving rank padding**

Add:

```julia
function pad_tt_ranks(
    x::TTTensor,
    ranks::AbstractVector{<:Integer},
)
    dims = tt_dims(x)
    old_ranks = collect(tt_ranks(x))
    requested = Int.(ranks)
    length(requested) == length(old_ranks) ||
        throw(ArgumentError("rank vector must have length $(length(old_ranks))"))
    requested[1] == 1 && requested[end] == 1 ||
        throw(ArgumentError("TT boundary ranks must be one"))
    all(requested .>= old_ranks) ||
        throw(ArgumentError("rank padding cannot shrink existing ranks"))

    limits = admissible_tt_ranks(dims, maximum(requested))
    all(requested .<= limits) ||
        throw(ArgumentError("requested ranks exceed Hilbert-space bounds"))

    T = promote_type(map(core -> eltype(core), x.cores)...)
    cores = Vector{Array{T,3}}(undef, length(x.cores))
    for i in eachindex(x.cores)
        old = x.cores[i]
        r1, n, r2 = size(old)
        core = zeros(T, requested[i], n, requested[i + 1])
        core[1:r1, :, 1:r2] .= old
        cores[i] = core
    end
    return TTTensor(cores)
end
```

- [ ] **Step 5: Implement first-site populations**

Add:

```julia
function first_site_populations(psi::TTTensor)
    isempty(psi.cores) && throw(ArgumentError("TT state must contain a core"))

    T = promote_type(ComplexF64, map(core -> eltype(core), psi.cores)...)
    env = ones(T, 1, 1)
    for site in length(psi.cores):-1:2
        core = T.(psi.cores[site])
        r1, n, r2 = size(core)
        next_env = zeros(T, r1, r1)
        for a in 1:r1, b in 1:r1, state in 1:n, c in 1:r2, d in 1:r2
            next_env[a, b] +=
                core[a, state, c] * conj(core[b, state, d]) * env[c, d]
        end
        env = next_env
    end

    first_core = T.(psi.cores[1])
    _, nsys, r2 = size(first_core)
    populations = zeros(Float64, nsys)
    for state in 1:nsys, c in 1:r2, d in 1:r2
        populations[state] += real(
            first_core[1, state, c] *
            conj(first_core[1, state, d]) *
            env[c, d],
        )
    end
    return populations
end
```

- [ ] **Step 6: Run utility tests and verify GREEN**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: all utility assertions and the pre-existing package tests pass.

- [ ] **Step 7: Add a small KSL norm-preservation test**

Append:

```julia
@testset "KSL padded-state evolution" begin
    sigma_x = ComplexF64[0 1; 1 0]
    number = ComplexF64[0 0; 0 1]
    identity2 = Matrix{ComplexF64}(I, 2, 2)
    H = tt_mkron(sigma_x, identity2) + 0.25 * tt_mkron(identity2, number)
    x0 = product_tt_state([
        ComplexF64[1, 0],
        ComplexF64[1, 0],
    ])
    ranks = admissible_tt_ranks(tt_dims(x0), 30)
    x0 = pad_tt_ranks(x0, ranks)
    x1 = tt_ksl(x0, 1im * H, 0.1; symmetric=true, rmax=30)

    @test norm(x1) ≈ norm(x0) rtol=1e-11
end
```

- [ ] **Step 8: Run the complete test suite and commit Task 1**

Run:

```bash
julia --project=. test/runtests.jl
git diff --check
```

Expected: all tests pass and the formatting check prints nothing.

Commit:

```bash
git add examples/tfd/ksl/utils.jl test/runtests.jl
git commit -m "feat: add KSL example TT utilities"
```

---

### Task 2: Ohmic spin-boson KSL/HEOM comparison

**Files:**
- Create: `examples/tfd/ksl/sb_ohmic.jl`

**Interfaces:**
- Consumes:
  - `admissible_tt_ranks`, `pad_tt_ranks`, and `first_site_populations` from Task 1.
  - `BosonicTFD`, `BosonicEnv`, `SBSystem`, `tt_sbham`, and `initial_tt_state` from TTDynamics.
  - `tt_ksl` from TTSolver.
- Produces: an executable example and `ksl_populations.csv`, `sb_ohmic_comparison.png`, `sb_ohmic_norm.png`, and `sb_ohmic_rank.png` in its working directory.

- [ ] **Step 1: Create imports, parameters, and shared utility loading**

Create `examples/tfd/ksl/sb_ohmic.jl` beginning with:

```julia
using LinearAlgebra
using DelimitedFiles
using TTSolver
using TTDynamics
using QFiND
using HEOMKit
using HEOMKit: icm2ifs
using ExpFit
using CairoMakie

include("utils.jl")

s = 1.0
gamma_c = 50.0
lambda = 5.0
temperature = 300.0
epsilon = 0.0
delta = 20.0
hierarchy_depth = 6

omega_min = -250.0
omega_max = 300.0
n_omega = 1000
identification_tmax = 500.0
n_identification_times = 200
identification_tolerance = 3e-2

esprit_tmax = 500.0
n_esprit_samples = 200
esprit_tolerance = 1e-3

t_end = 500.0
dt_heom = 0.5
dt_ksl = 2.0
rmax = 30
nb_min = 12
nb_max = 50
```

- [ ] **Step 2: Implement QFiND discretization and BCF diagnostics**

Use these concrete calls:

```julia
spectral_density = PowerLawExpSD(s, gamma_c; reorgene=lambda)
bcf = BosonicBCF(spectral_density, temperature; ub=1000.0)
quantum_spectral_density = BosonicQNSD(spectral_density, temperature)
identification_bcf = BosonicBCF(spectral_density, temperature; ub=10000.0)

dataset, _ = InitialData(
    DiscrID(),
    quantum_spectral_density,
    identification_bcf,
    omega_min,
    omega_max,
    identification_tmax;
    n_freq=n_omega,
    n_time=n_identification_times,
)
discretization = id_discr(dataset, identification_tolerance)
omega_bath = discretization.freq
g_bath = discretization.coeff
length(omega_bath) == length(g_bath) ||
    error("bath frequency and coupling counts differ")

bcf_qfind = bcf_approx.(dataset.time, Ref(omega_bath), Ref(g_bath))
qfind_error = norm(bcf_qfind - dataset.bcf) / norm(dataset.bcf)
```

Print the mode count and `qfind_error`.

- [ ] **Step 3: Implement the HEOM reference**

Use:

```julia
H_heom = [epsilon / 2 delta; delta -epsilon / 2] * icm2ifs
sample_times = range(0.0, esprit_tmax; length=n_esprit_samples)
sample_dt = step(sample_times)
bcf_samples = bcf.(sample_times)
fit = ExpFit.esprit(bcf_samples, sample_dt, esprit_tolerance)
fit_error = norm(fit.(sample_times) - bcf_samples) / norm(bcf_samples)

coupling_operator = ComplexF64[1 0; 0 -1]
bath = BathExp(fit.expon, fit.coeff, coupling_operator)
system = HEOMSystem(H_heom, NoiseExp(bath), hierarchy_depth; hierarchy=:depth)
initial_ados = initial_ado(system, 1)
times_heom, populations_heom =
    evolve(system, initial_ados, (0.0, t_end), dt_heom; parallel=true)
```

Print the ESPRIT term count and `fit_error`.

- [ ] **Step 4: Build the matching TFD Hamiltonian and padded state**

Use:

```julia
omega_fs = omega_bath .* icm2ifs
g_fs = g_bath .* icm2ifs
sigma_z = Matrix(PAULI.σz)
sigma_x = Matrix(PAULI.σx)
H_system =
    0.5 * (epsilon * icm2ifs) * sigma_z +
    (delta * icm2ifs) * sigma_x

tfd_bath = BosonicTFD(omega_fs, g_fs, ComplexF64.(sigma_z))
sb_system = SBSystem(H_system, BosonicEnv(tfd_bath))
H_tt, basis_sizes = tt_sbham(
    sb_system;
    threshold=0.9999,
    nb_min=nb_min,
    nb_max=nb_max,
)

spaces = vcat(
    [SpinSpace(0.5)],
    [BosonSpace(n) for n in basis_sizes],
)
localized = initial_tt_state(spaces; state_index=1, site=1)
target_ranks = admissible_tt_ranks(tt_dims(localized), rmax)
u = pad_tt_ranks(localized, target_ranks)
```

Print `basis_sizes`, `target_ranks`, the total Hilbert-space dimension, and
the norm before evolution.

- [ ] **Step 5: Implement symmetric KSL evolution and recording**

Use:

```julia
times_ksl = collect(0.0:dt_ksl:t_end)
norms_ksl = zeros(Float64, length(times_ksl))
populations_ksl = zeros(Float64, 2, length(times_ksl))
rank_history = Vector{Vector{Int}}(undef, length(times_ksl))

norms_ksl[1] = real(dot(u, u))
populations_ksl[:, 1] .= first_site_populations(u)
rank_history[1] = collect(tt_ranks(u))

A = 1im * H_tt
for step_index in 2:length(times_ksl)
    u = tt_ksl(
        u,
        A,
        dt_ksl;
        symmetric=true,
        rmax=rmax,
        tol=1e-12,
    )
    norms_ksl[step_index] = real(dot(u, u))
    populations_ksl[:, step_index] .= first_site_populations(u)
    rank_history[step_index] = collect(tt_ranks(u))
    println(
        step_index,
        "  t=", times_ksl[step_index],
        "  P=", populations_ksl[:, step_index],
        "  norm=", norms_ksl[step_index],
        "  rmax=", maximum(rank_history[step_index]),
    )
end
```

- [ ] **Step 6: Save CSV data**

Write:

```julia
open("ksl_populations.csv", "w") do io
    writedlm(
        io,
        ["time_fs" "population_up" "population_down" "norm" "max_rank"],
        ',',
    )
    rows = hcat(
        times_ksl,
        vec(populations_ksl[1, :]),
        vec(populations_ksl[2, :]),
        norms_ksl,
        maximum.(rank_history),
    )
    writedlm(io, rows, ',')
end
```

- [ ] **Step 7: Create comparison, norm, and rank figures**

Create three CairoMakie figures:

```julia
comparison = Figure(size=(900, 600))
axis = Axis(
    comparison[1, 1];
    xlabel="Time [fs]",
    ylabel="Population",
    title="Spin-Boson with Ohmic Bath: KSL vs HEOM",
)
lines!(axis, times_heom, real.(populations_heom[1, :]); label="HEOM rho11")
lines!(axis, times_heom, real.(populations_heom[2, :]); label="HEOM rho22")
lines!(axis, times_ksl, populations_ksl[1, :]; label="KSL P(up)", linestyle=:dash)
lines!(axis, times_ksl, populations_ksl[2, :]; label="KSL P(down)", linestyle=:dash)
axislegend(axis; position=:rt)
save("sb_ohmic_comparison.png", comparison)
```

The norm figure plots `norms_ksl` and a horizontal line at one. The rank
figure plots every internal bond rank in `rank_history` against `times_ksl`
and also plots `maximum.(rank_history)` with a thicker black line. Save them
as the filenames required by the interface block.

- [ ] **Step 8: Run lightweight verification**

Run:

```bash
julia --project=. test/runtests.jl
julia --project=. -e '
source = read("examples/tfd/ksl/sb_ohmic.jl", String)
Meta.parseall(source)
println("example syntax OK")
'
git diff --check
```

Expected: package tests pass, parsing prints `example syntax OK`, and the
formatting check prints nothing. Do not run the 500 fs simulation as part of
this task unless explicitly requested.

- [ ] **Step 9: Confirm the example does not modify existing dirty files**

Run:

```bash
git status --short
git diff --name-only HEAD
```

Expected task-owned changes are only:

```text
examples/tfd/ksl/sb_ohmic.jl
```

Task 1 files are already committed. Pre-existing user changes outside the
isolated implementation workspace must not appear in the task diff.

- [ ] **Step 10: Commit Task 2**

```bash
git add examples/tfd/ksl/sb_ohmic.jl
git commit -m "feat: add KSL Ohmic spin-boson example"
```
