# Spin-Boson model simulation using tAMEn
#
# This example demonstrates the tAMEn (Time-dependent AMEn) integrator for
# time evolution of quantum systems in TT format with rank adaptation.

using LinearAlgebra
using TTSolver
using TTDynamics

include("../common/plot_utils.jl")

# Parameters
nb = 6
nt = 3
tol = 1e-4
Tfin = 10.0
dt = 0.05
Nsteps = 200

# Create physical spaces
spin = SpinSpace(0.5)
boson = BosonSpace(nb)

# Get operators
σz = Matrix(PAULI.σz)
σx = Matrix(PAULI.σx)
id2 = Matrix(PAULI.I)
x_op = Matrix(boson.a + boson.adag)  # x = a + a†
n_op = Matrix(boson.n)
idb = Matrix{ComplexF64}(I, nb, nb)

Δ0 = 1.0
Δx = 1.5
ω = 1.2
g = 0.3

# Hamiltonian in TT format
H_tt =
    0.5 * Δ0 * tt_mkron(σz, idb) +
    0.5 * Δx * tt_mkron(σx, idb) +
    ω * tt_mkron(id2, n_op) +
    g * tt_mkron(σz, x_op)

# Full Hamiltonian for exact solution
H_full =
    0.5 * Δ0 * kron(σz, idb) +
    0.5 * Δx * kron(σx, idb) +
    ω * kron(id2, n_op) +
    g * kron(σz, x_op)

opts = Dict(
    :verb => 1,
    :nswp => 2,
    :local_iters => 5,
    :time_scheme => "cn",
    :time_error_damp => 10.0,
    :kickrank => 4,
    :correct_2_norm => true,
)

# Initial state: |↑, 0⟩
x0 = initial_tt_state([spin, boson], state_index=1, site=1)
U = tkron(x0, tt_ones(nt; T=ComplexF64))

A = (-1im) * H_tt

ts = collect(range(0.0, Tfin; length=Nsteps + 1))

psi0 = kron(ComplexF64[1.0, 0.0], ComplexF64[1.0; zeros(ComplexF64, nb - 1)])

# Storage for results
psi_exact = Vector{Vector{ComplexF64}}(undef, length(ts))
psi_tt = Vector{Vector{ComplexF64}}(undef, length(ts))
norm_exact = zeros(Float64, length(ts))
norm_tt = zeros(Float64, length(ts))

psi_exact[1] = psi0
psi_tt[1] = vec(tt_full(x0))
norm_exact[1] = sum(abs2, psi_exact[1])
norm_tt[1] = sum(abs2, psi_tt[1])

println("Running tAMEn time evolution...")
println("Parameters: nb=$nb, nt=$nt, dt=$dt, Tfin=$Tfin, g=$g")

scheme = String(opts[:time_scheme])
let u = x0
    for i in 2:length(ts)
        A_step = dt * A
        U = tkron(u, tt_ones(nt; T=ComplexF64))
        U, tgrid, _, _ = tamen(U, A_step, tol, opts)
        u = extract_snapshot(U, tgrid, 1.0, scheme)
        norm_tt[i] = real(dot(u, u))
        psi_tt[i] = vec(tt_full(u))
        psi_exact[i] = exp(-1im * H_full * ts[i]) * psi0
        norm_exact[i] = sum(abs2, psi_exact[i])
    end
end

# Compute populations
p_up_exact, p_dn_exact = compute_spin_populations(psi_exact, nb)
p_up_tt, p_dn_tt = compute_spin_populations(psi_tt, nb)

# Compute errors
max_error_up = maximum(abs.(p_up_exact - p_up_tt))
max_error_dn = maximum(abs.(p_dn_exact - p_dn_tt))
println("\nResults:")
println("  Max error P(↑): $max_error_up")
println("  Max error P(↓): $max_error_dn")
println("  Max norm deviation: $(maximum(abs.(norm_tt .- 1.0)))")

# Plot populations
fig_pop = plot_spin_boson_populations(
    ts, p_up_exact, p_up_tt, p_dn_exact, p_dn_tt;
    method="tAMEn",
    title_suffix="tAMEn, g=$g",
    filename="sb_tamen.png"
)
display(fig_pop)

# Plot norm
fig_norm = plot_norm_conservation(
    ts, norm_exact, norm_tt;
    method="tAMEn",
    filename="sb_tamen_norm.png"
)
display(fig_norm)

# Save data
save_spin_boson_data("sb_tamen_data.csv", ts, p_up_exact, p_up_tt, p_dn_exact, p_dn_tt, norm_exact, norm_tt)
