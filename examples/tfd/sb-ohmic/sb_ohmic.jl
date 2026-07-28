using LinearAlgebra
using TTSolver
using TTDynamics
using QFiND
using KaisouEOM
using KaisouEOM: icm2ifs, kB
using ExpFit
using CairoMakie

include("../../common/plot_utils.jl")

println("=" ^ 60)
println("Spin-Boson with Ohmic Bath: tAMEn vs HEOM comparison")
println("=" ^ 60)

# =============================================
# Spectral Density Parameters (Ohmic bath)
# =============================================
s = 1.0          # Ohmic (s=1), sub-Ohmic (s<1), super-Ohmic (s>1)
γc = 50.0        # Cutoff frequency [cm⁻¹]
λ = 5.0          # Reorganization energy [cm⁻¹]
Temp = 300.0     # Temperature [K]

# System Hamiltonian parameters
ε = 0.0          # Energy difference [cm⁻¹]
Δ = 20.0         # Tunneling coupling [cm⁻¹]

# =============================================
# QFiND: Discretize spectral density to get ω, g
# =============================================
println("\n" * "=" ^ 60)
println("QFiND: Computing bath frequencies and couplings...")
println("=" ^ 60)

# Create spectral density
sdens = PowerLawExpSD(s, γc; reorgene=λ)
bcf_func = BosonicBCF(sdens, Temp; ub=1000.0)

# QFiND parameters
Ω_min = -250.0
Ω_max = 300.0
N_ω = 1000
T_max = 150.0    # [fs] - time range for BCF fitting
N_t = 100
eps_id = 3e-2    # Discretization tolerance

# ESPRIT fitting for BCF
tmin = 0.0
tmax_esprit = 150.0   # [fs]
nsamples = 100
eps_esprit = 1e-3

# Time evolution parameters
t_end = 150.0    # [fs]
dt_heom = 0.5    # [fs]

# Initialize data for ID method
sbeta = BosonicQNSD(sdens, Temp)
bcf_for_id = BosonicBCF(sdens, Temp; ub=10000.0)
dataset, _ = InitialData(DiscrID(), sbeta, bcf_for_id, Ω_min, Ω_max, T_max; n_freq=N_ω, n_time=N_t)

# Run identification discretization
res = id_discr(dataset, eps_id)
ω_bath = res.freq  # Frequencies [cm⁻¹]
g_bath = res.coeff # Coupling coefficients

n_modes = length(ω_bath)

# Check BCF approximation quality
t_check = dataset.time
bcf_approx_vals = bcf_approx.(t_check, Ref(ω_bath), Ref(g_bath))
bcf_error = norm(bcf_approx_vals .- dataset.bcf) / norm(dataset.bcf)
println("  BCF approximation error: $(round(bcf_error, sigdigits=3))")

# =============================================
# HEOM using KaisouEOM (reference solution)
# =============================================
println("\n" * "=" ^ 60)
println("HEOM: Running reference dynamics with KaisouEOM...")
println("=" ^ 60)

# System Hamiltonian in HEOM units
H_heom = [ε/2 Δ; Δ -ε/2] * icm2ifs  # Convert to [1/fs]

dt_sample = (tmax_esprit - tmin) / (nsamples - 1)
t_samples = range(tmin, tmax_esprit, length=nsamples)
bcf_samples = [bcf_func(t) for t in t_samples]

# ESPRIT fitting
ef = ExpFit.esprit(bcf_samples, dt_sample, eps_esprit)
println("  ESPRIT fitting: $(length(ef.expon)) exponential terms")

bcf_fit = [ef(t) for t in t_samples]
fit_error_esprit = norm(bcf_fit .- bcf_samples) / norm(bcf_samples)
println("  ESPRIT fitting error: $(round(fit_error_esprit, sigdigits=3))")

# Build bath from ESPRIT results
V = ComplexF64[1 0; 0 -1]  # σz coupling
bath = BathExp(ef.expon, ef.coeff, V; add_conjugate=true)
noise = NoiseExp(bath)

# HEOM system
ndepth = 6
system = HEOMSystem(H_heom, noise, ndepth; hierarchy=:depth)
# Initial condition: localized on state |1⟩
P0 = initial_ado(system, 1)

# Run HEOM dynamics
println("  Running HEOM dynamics...")
times_heom, pops_heom = evolve(system, P0, (0.0, t_end), dt_heom; parallel=true)
println("  Done!")

# =============================================
# tAMEn dynamics using TTDynamics types
# =============================================
println("\n" * "=" ^ 60)
println("tAMEn: Running TT-based dynamics using TTDynamics...")
println("=" ^ 60)

# tAMEn parameters
nb_default = 12  # Default boson truncation per mode
nb_max = 50      # Maximum boson truncation
nt = 3          
tol = 7e-3
Tfin_fs = t_end  # [fs] - same as HEOM
dt_tamen = 2.0   # [fs]
Nsteps = Int(Tfin_fs / dt_tamen)

# Convert units: cm⁻¹ → fs⁻¹
ω_fs = ω_bath .* icm2ifs  # Frequencies in fs⁻¹
g_fs = g_bath .* icm2ifs  # Couplings in sqrt(fs⁻¹)

# System Hamiltonian [fs⁻¹]
ε_fs = ε * icm2ifs
Δ_fs = Δ * icm2ifs
σz = Matrix(PAULI.σz)
σx = Matrix(PAULI.σx)
Hsys = 0.5 * ε_fs * σz + Δ_fs * σx

# Coupling operator
V_coupling = ComplexF64.(σz)
# Create BosonicTFD for the Ohmic bath
bath_tfd = BosonicTFD(ω_fs, g_fs, V_coupling)
# Create BosonicEnv
benv = BosonicEnv(bath_tfd)
# Create SBSystem
sbsys = SBSystem(Hsys, benv)
# Build Hamiltonian in TT format with automatic basis size estimation
H_tt, basis_sizes = tt_sbham(sbsys; threshold=0.9999, nb_default=nb_default, nb_max=nb_max)

# Total Hilbert space dimension
nb_total = prod(basis_sizes)
println("  Total boson Hilbert space: $nb_total (= $(join(basis_sizes, "×")))")
println("  Total Hilbert space: $(2 * nb_total)")

# Create spaces for initial state
spin = SpinSpace(0.5)
boson_spaces = [BosonSpace(basis_sizes[i]) for i in 1:n_modes]
spaces = vcat([spin], boson_spaces)
dims = [dim(s) for s in spaces]  # Dimensions for tt_partial_trace

opts = Dict(:verb => 0, :nswp => 2, :local_iters => 5, :time_scheme => "cn", :time_error_damp => 100.0, :kickrank => 4, :correct_2_norm => true,)

# Initial state: |↑, 0, 0, ...⟩
x0 = initial_tt_state(spaces, state_index=1, site=1)
# Time grid
ts_tamen = collect(range(0.0, Tfin_fs; length=Nsteps + 1))
# Storage for tAMEn results
psi_tt = Vector{Vector{ComplexF64}}(undef, length(ts_tamen))
norm_tt = zeros(Float64, length(ts_tamen))

psi_tt[1] = vec(tt_full(x0))
norm_tt[1] = sum(abs2, psi_tt[1])

A = (-1im) * H_tt

println("  Running tAMEn time evolution...")
println("  Parameters: basis=$(basis_sizes), nt=$nt, dt=$dt_tamen fs, Tfin=$Tfin_fs fs")
println()
println("  Step |   Time [fs]  |    P(↑)    |    P(↓)    |   Norm")
println("  " * "-"^60)

# Print initial state using system_populations
pops_init = system_populations(psi_tt[1], 1, dims)
println("  $(lpad(1, 4)) |  $(lpad(round(ts_tamen[1], digits=2), 9))  |  $(lpad(round(pops_init[1], digits=6), 8))  |  $(lpad(round(pops_init[2], digits=6), 8))  |  $(round(norm_tt[1], digits=6))")

scheme = String(opts[:time_scheme])
let u = x0
    for i in 2:length(ts_tamen)
        A_step = dt_tamen * A
        U = tkron(u, tt_ones(nt; T=ComplexF64))
        U, tgrid, _, _ = tamen(U, A_step, tol, opts)
        u = extract_snapshot(U, tgrid, 1.0, scheme)
        norm_tt[i] = real(dot(u, u))
        psi_tt[i] = vec(tt_full(u))
        
        # Print populations using system_populations
        pops = system_populations(psi_tt[i], 1, dims)
        println("  $(lpad(i, 4)) |  $(lpad(round(ts_tamen[i], digits=2), 9))  |  $(lpad(round(pops[1], digits=6), 8))  |  $(lpad(round(pops[2], digits=6), 8))  |  $(round(norm_tt[i], digits=6))")
    end
end
println("  " * "-"^60)
println("  Done!")

# Compute populations for tAMEn using system_populations
pops_tt = system_populations(psi_tt, 1, dims)
p_up_tt = pops_tt[1, :]
p_dn_tt = pops_tt[2, :]

println("\ntAMEn Results:")
println("  Max norm deviation: $(maximum(abs.(norm_tt .- 1.0)))")

# =============================================
# 5. Comparison Plots
# =============================================
println("\n" * "=" ^ 60)
println("Generating comparison plots...")
println("=" ^ 60)

# Figure 1: Population dynamics comparison
fig1 = Figure(size=(900, 600))
ax1 = Axis(fig1[1, 1],
    xlabel = "Time [fs]",
    ylabel = "Population",
    title = "Spin-Boson with Ohmic Bath: tAMEn vs HEOM\n(λ=$λ cm⁻¹, γ=$γc cm⁻¹, T=$Temp K, Δ=$Δ cm⁻¹)"
)

# HEOM results
lines!(ax1, times_heom, real.(pops_heom[1, :]), linewidth=3, label="HEOM ρ₁₁", color=:blue)
lines!(ax1, times_heom, real.(pops_heom[2, :]), linewidth=3, label="HEOM ρ₂₂", color=:red)

# tAMEn results
lines!(ax1, ts_tamen, p_up_tt, linewidth=2, linestyle=:dash, label="tAMEn P(↑)", color=:cyan)
lines!(ax1, ts_tamen, p_dn_tt, linewidth=2, linestyle=:dash, label="tAMEn P(↓)", color=:orange)

axislegend(ax1, position=:rt)

save("sb_ohmic_comparison.png", fig1)
println("  Saved: sb_ohmic_comparison.png")
display(fig1)

# Figure 2: BCF fitting quality
fig2 = Figure(size=(800, 500))
ax2 = Axis(fig2[1, 1],
    xlabel = "Time [fs]",
    ylabel = "C(t)",
    title = "Bath Correlation Function: QFiND vs ESPRIT"
)

# Plot BCF
t_plot = collect(range(0.0, 150.0, length=150))
bcf_exact = [bcf_func(t) for t in t_plot]
bcf_qfind = [bcf_approx(t, ω_bath, g_bath) for t in t_plot]
bcf_esprit = [ef(t) for t in t_plot]

lines!(ax2, t_plot, real.(bcf_exact), linewidth=2, label="Re[C(t)] Exact", color=:black)
lines!(ax2, t_plot, real.(bcf_qfind), linewidth=2, linestyle=:dash, label="Re[C(t)] QFiND", color=:blue)
lines!(ax2, t_plot, real.(bcf_esprit), linewidth=2, linestyle=:dot, label="Re[C(t)] ESPRIT", color=:red)

lines!(ax2, t_plot, imag.(bcf_exact), linewidth=2, label="Im[C(t)] Exact", color=:gray)
lines!(ax2, t_plot, imag.(bcf_qfind), linewidth=2, linestyle=:dash, label="Im[C(t)] QFiND", color=:cyan)
lines!(ax2, t_plot, imag.(bcf_esprit), linewidth=2, linestyle=:dot, label="Im[C(t)] ESPRIT", color=:orange)

axislegend(ax2, position=:rt)

save("bcf_comparison.png", fig2)
println("  Saved: bcf_comparison.png")
display(fig2)

# Figure 3: Norm conservation for tAMEn
fig3 = Figure(size=(600, 400))
ax3 = Axis(fig3[1, 1],
    xlabel = "Time [fs]",
    ylabel = "Norm",
    title = "tAMEn Norm Conservation"
)
lines!(ax3, ts_tamen, norm_tt, linewidth=2, color=:blue)
hlines!(ax3, [1.0], linewidth=1, linestyle=:dash, color=:gray)

save("sb_ohmic_norm.png", fig3)
println("  Saved: sb_ohmic_norm.png")
display(fig3)

# Figure 4: Discretized bath modes
fig4 = Figure(size=(800, 500))
ax4 = Axis(fig4[1, 1],
    xlabel = "Frequency ω [cm⁻¹]",
    ylabel = "|g|² [cm⁻¹]",
    title = "Discretized Bath Modes (QFiND)"
)

# =============================================
# 6. Save data
# =============================================
println("\n" * "=" ^ 60)
println("Saving data...")
println("=" ^ 60)

println("\n" * "=" ^ 60)
println("Simulation completed!")
println("=" ^ 60)
