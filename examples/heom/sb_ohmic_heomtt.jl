# Spin-Boson model with Ohmic bath: HEOM-TT vs standard HEOM comparison
#
# This example demonstrates:
# 1. HEOM (Hierarchical Equations of Motion) using HEOMKit
# 2. HEOM-TT: HEOM equations solved in Tensor-Train format using tAMEn
#
# Reference: Borrelli, Gelin, Chem. Phys. 481 (2016) 91-98
#
# Hamiltonian:
#   H = ε/2 σz + Δ/2 σx + bath
# System-bath coupling: V = σz

using LinearAlgebra
using TTSolver
using TTDynamics
using QFiND
using HEOMKit
using HEOMKit: icm2ifs
using ExpFit
using CairoMakie

include("../common/plot_utils.jl")

function main()
println("=" ^ 60)
println("Spin-Boson with Ohmic Bath: HEOM-TT vs standard HEOM")
println("=" ^ 60)

# =============================================
# 1. Spectral Density Parameters (Ohmic/Drude bath)
# =============================================
s = 1.0          # Ohmic (s=1)
γc = 50.0        # Cutoff frequency [cm⁻¹]
λ = 5.0         # Reorganization energy [cm⁻¹]
Temp = 300.0     # Temperature [K]

# System Hamiltonian parameters
ε = 0.0          # Energy difference [cm⁻¹]
Δ = 20.0        # Tunneling coupling [cm⁻¹]
nsys = 2         # Number of system states

# =============================================
# 2. ESPRIT: Fit BCF to sum of exponentials
# =============================================
println("\n" * "=" ^ 60)
println("ESPRIT: Fitting BCF to sum of exponentials...")
println("=" ^ 60)

# Create spectral density and BCF
sdens = PowerLawExpSD(s, γc; reorgene=λ)
bcf_func = BosonicBCF(sdens, Temp; ub=1000.0)

# ESPRIT fitting parameters
tmin = 0.0
tmax_esprit = 500.0   # [fs]
nsamples = 100
eps_esprit = 1e-2   # Smaller eps → more exponential terms

# Time evolution parameters
t_end = 500.0    # [fs]
dt_heom = 0.5    # [fs]

dt_sample = (tmax_esprit - tmin) / (nsamples - 1)
t_samples = range(tmin, tmax_esprit, length=nsamples)
bcf_samples = [bcf_func(t) for t in t_samples]

# ESPRIT fitting
ef = ExpFit.esprit(bcf_samples, dt_sample, eps_esprit)
n_bcf = length(ef.expon)
println("  ESPRIT fitting: $n_bcf exponential terms")

# Get exponential terms: C(t) = Σₖ cₖ exp(γₖ t)  (γ < 0 for decay)
γ_bcf = ef.expon  # in fs⁻¹
c_bcf = ef.coeff  # in (cm⁻¹)²

println("  Exponential terms:")
for k in 1:n_bcf
    println("    k=$k: γ = $(round(γ_bcf[k], sigdigits=4)) fs⁻¹, c = $(round(c_bcf[k], sigdigits=4))")
end

# Verify BCF fitting quality
bcf_fit = [ef(t) for t in t_samples]
fit_error = norm(bcf_fit .- bcf_samples) / norm(bcf_samples)
println("  ESPRIT fitting error: $(round(fit_error, sigdigits=3))")

# =============================================
# 3. Standard HEOM using HEOMKit (reference)
# =============================================
println("\n" * "=" ^ 60)
println("Standard HEOM: Running with HEOMKit...")
println("=" ^ 60)

# System Hamiltonian in fs⁻¹ units
H_sys = [ε/2 Δ; Δ -ε/2] * icm2ifs

# Build bath from ESPRIT results
V = ComplexF64[1 0; 0 -1]  # σz coupling
bath = BathExp(ef.expon, ef.coeff, V)
noise = NoiseExp(bath)

# HEOM parameters (use depth hierarchy)
ndepth = 10
system = HEOMSystem(H_sys, noise, ndepth; hierarchy=:depth)

# Initial condition: localized on state |1⟩
P0 = initial_ado(system, 1)

# Run HEOM dynamics
println("  Running HEOM dynamics...")
times_heom, pops_heom = evolve(system, P0, (0.0, t_end), dt_heom; parallel=true)
println("  Done! $(length(times_heom)) time steps")

# =============================================
# 4. HEOM-TT: Build Liouvillian using TTDynamics
# =============================================
println("\n" * "=" ^ 60)
println("HEOM-TT: Building Liouvillian...")
println("=" ^ 60)

# HEOM-TT parameters
Nh = 10         # Maximum hierarchy depth per mode (match HEOM)
nt = 3           # Number of time points for tAMEn
tol = 1e-3       # TT tolerance
dt_tt = 1.0      # Time step [fs]
Nsteps = Int(t_end / dt_tt)

# Build HEOMTTSystem using NoiseExp (already constructed for HEOMKit HEOM)
params = HEOMTTSystem(H_sys, noise, Nh)
tt_dimensions = heom_tt_dimensions(params)
println("  HEOM-TT mode dimensions [ket, bra, hierarchy...]: $tt_dimensions")
println("  TT cores: 2 system cores + $(length(params.nb)) hierarchy cores = $(length(tt_dimensions))")

# Build HEOM Liouvillian
liouvillian, trace_observable, population_observables =
    build_heom_liouvillian(params; tol=1e-10)

# Build initial state (|1⟩⟨1|)
rho0_tt = build_initial_state(params, 1)
println("  Initial state TT-ranks: $(tt_ranks(rho0_tt))")

# =============================================
# 5. Time evolution with tAMEn
# =============================================
println("\n" * "=" ^ 60)
println("HEOM-TT: Running tAMEn time evolution...")
println("=" ^ 60)

time_scheme = "CN"
workspace = TAMEnWorkspace(ComplexF64)

# Storage for results
ts_tt = collect(range(0.0, t_end; length=Nsteps + 1))
pop_tt = zeros(Float64, nsys, length(ts_tt))
system_trace_tt = zeros(Float64, length(ts_tt))
trace_error_tt = zeros(Float64, length(ts_tt))
rank_history = Vector{Vector{Int}}(undef, length(ts_tt))

# Initial populations
for k in 1:nsys
    pop_tt[k, 1] = real(tt_dot(population_observables[k], rho0_tt))
end
system_trace_tt[1] = real(tt_dot(trace_observable, rho0_tt))
trace_error_tt[1] = abs(system_trace_tt[1] - 1.0)
rank_history[1] = collect(tt_ranks(rho0_tt))

println("  Parameters: Nh=$Nh, n_bcf=$n_bcf, nt=$nt, dt=$dt_tt fs, Tfin=$t_end fs")
println()
println("  Step | Time [fs] |   P₁₁   |   P₂₂   | system_trace | trace_error | MaxRank")
println("  " * "-"^86)
println("  $(lpad(0, 4)) | $(lpad(round(ts_tt[1], digits=1), 9)) | $(lpad(round(pop_tt[1,1], digits=5), 7)) | $(lpad(round(pop_tt[2,1], digits=5), 7)) | $(lpad(round(system_trace_tt[1], digits=8), 12)) | $(lpad(round(trace_error_tt[1], sigdigits=3), 11)) | $(maximum(rank_history[1]))")

# Scale the Liouvillian by the physical time step
A_step = dt_tt * liouvillian

u = rho0_tt
for i in 2:length(ts_tt)
    # tAMEn step
    space_time = tt_kron(u, tt_ones(nt; T=ComplexF64))
    solution, time_grid, _, _, _ = tamen(
        space_time,
        A_step,
        tol;
        verb=0,
        nswp=5,
        local_iters=10,
        time_scheme,
        time_error_damp=100.0,
        kickrank=4,
        correct_2_norm=false,
        conserved_observables=[trace_observable],
        workspace,
    )
    u = extract_snapshot(solution, time_grid, 1.0, time_scheme)

    # Compute observables
    for k in 1:nsys
        pop_tt[k, i] = real(tt_dot(population_observables[k], u))
    end
    system_trace_tt[i] = real(tt_dot(trace_observable, u))
    trace_error_tt[i] = abs(system_trace_tt[i] - 1.0)
    rank_history[i] = collect(tt_ranks(u))

    # Print progress every 10 steps
    if i % 10 == 0 || i == length(ts_tt)
        println("  $(lpad(i-1, 4)) | $(lpad(round(ts_tt[i], digits=1), 9)) | $(lpad(round(pop_tt[1,i], digits=5), 7)) | $(lpad(round(pop_tt[2,i], digits=5), 7)) | $(lpad(round(system_trace_tt[i], digits=8), 12)) | $(lpad(round(trace_error_tt[i], sigdigits=3), 11)) | $(maximum(rank_history[i]))")
    end
end
println("  " * "-"^86)
println("  Done!")

# =============================================
# 6. Comparison Plots
# =============================================
println("\n" * "=" ^ 60)
println("Generating comparison plots...")
println("=" ^ 60)

# Population dynamics comparison
plot_spin_boson_comparison(
    times_heom,
    pops_heom,
    ts_tt,
    pop_tt;
    reference_method="HEOM",
    method="HEOM-TT",
    xlabel="Time [fs]",
    title="Spin-Boson Ohmic Bath: HEOM-TT vs Standard HEOM\n(λ=$λ cm⁻¹, γ=$γc cm⁻¹, T=$Temp K, Δ=$Δ cm⁻¹)",
    filename="sb_ohmic_heomtt_comparison.png",
)

# Physical-system trace conservation
plot_norm_conservation(
    ts_tt,
    system_trace_tt;
    method="HEOM-TT",
    target=1.0,
    quantity="Tr[ρ_sys]",
    xlabel="Time [fs]",
    title="HEOM-TT System Trace Conservation Error",
    filename="sb_ohmic_heomtt_norm.png",
)

# TT-rank evolution
plot_tt_rank_history(
    ts_tt,
    rank_history;
    method="HEOM-TT",
    xlabel="Time [fs]",
    title="HEOM-TT Rank Evolution",
    filename="sb_ohmic_heomtt_rank.png",
)

# BCF fitting
fig4 = Figure(size=(800, 500))
ax4 = Axis(fig4[1, 1],
    xlabel = "Time [fs]",
    ylabel = "C(t)",
    title = "Bath Correlation Function (ESPRIT fit, $n_bcf terms)"
)

t_plot = collect(range(0.0, tmax_esprit, length=200))
bcf_exact = [bcf_func(t) for t in t_plot]
bcf_fit_plot = [ef(t) for t in t_plot]

lines!(ax4, t_plot, real.(bcf_exact), linewidth=2, label="Re[C(t)] Exact", color=:blue)
lines!(ax4, t_plot, real.(bcf_fit_plot), linewidth=2, linestyle=:dash, label="Re[C(t)] ESPRIT", color=:cyan)
lines!(ax4, t_plot, imag.(bcf_exact), linewidth=2, label="Im[C(t)] Exact", color=:red)
lines!(ax4, t_plot, imag.(bcf_fit_plot), linewidth=2, linestyle=:dash, label="Im[C(t)] ESPRIT", color=:orange)

axislegend(ax4, position=:rt)

save("sb_ohmic_bcf_esprit.png", fig4)
println("  Saved: sb_ohmic_bcf_esprit.png")
display(fig4)

# =============================================
# 7. Summary
# =============================================
println("\n" * "=" ^ 60)
println("Summary")
println("=" ^ 60)

# Print detailed comparison
println("  Time comparison (HEOM vs HEOM-TT):")
println("   t[fs]  |  HEOM ρ₁₁  |  TT ρ₁₁   |   Diff")
println("  " * "-"^45)
for i in 1:10:min(51, length(times_heom))
    t = times_heom[i]
    heom_val = real(pops_heom[1, i])
    tt_idx = findfirst(x -> isapprox(x, t; atol=0.1), ts_tt)
    if !isnothing(tt_idx)
        tt_val = pop_tt[1, tt_idx]
        diff = heom_val - tt_val
        println("   $(lpad(round(t, digits=1), 5))  |  $(lpad(round(heom_val, digits=5), 8))  |  $(lpad(round(tt_val, digits=5), 8))  |  $(round(diff, sigdigits=2))")
    end
end
println()

println("  Standard HEOM:")
println("    - Hierarchy depth: $ndepth")
println("    - Time steps: $(length(times_heom))")
println()
println("  HEOM-TT:")
println("    - BCF exponential terms: $n_bcf")
println("    - Hierarchy truncation per mode: $Nh")
println("    - Time steps: $(length(ts_tt))")
println("    - Final max TT-rank: $(maximum(rank_history[end]))")
println("    - Maximum system trace error: $(maximum(trace_error_tt))")
println()
println("Simulation completed!")
println("=" ^ 60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
