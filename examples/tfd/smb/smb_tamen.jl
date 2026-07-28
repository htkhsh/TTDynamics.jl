# Spin-Multi-Boson model simulation using tAMEn
#
# This example demonstrates the tAMEn (Time-dependent AMEn) integrator for
# time evolution of quantum systems in TT format with rank adaptation.
# The model consists of a spin coupled to three independent boson modes.
#
# Hamiltonian:
#   H = Δ0/2 σz + Δx/2 σx + Σᵢ ωᵢ nᵢ + g σz Σᵢ xᵢ

using LinearAlgebra
using TTSolver
using TTDynamics

include("../common/plot_utils.jl")

# Parameters
nb = 6          # Boson truncation per mode
n_bosons = 3    # Number of boson modes
nt = 3          # Time TT rank
tol = 1e-4
Tfin = 10.0
dt = 0.05
Nsteps = 200

# Create physical spaces
spin = SpinSpace(0.5)
boson = BosonSpace(nb)
spaces = [spin, boson, boson, boson]  # spin + 3 bosons

# Get operators
σz = Matrix(PAULI.σz)
σx = Matrix(PAULI.σx)
id2 = Matrix(PAULI.I)
x_op = Matrix(boson.a + boson.adag)  # x = a + a†
n_op = Matrix(boson.n)
idb = Matrix{ComplexF64}(I, nb, nb)

# Physical parameters
Δ0 = 1.0      # σz coefficient
Δx = 1.5      # σx coefficient
ω = [1.2, 1.0, 0.8]  # Boson frequencies (can be different)
g = 0.3       # Spin-boson coupling

# Build Hamiltonian in TT format for spin + 3 independent bosons
# H = Δ0/2 σz ⊗ I ⊗ I ⊗ I + Δx/2 σx ⊗ I ⊗ I ⊗ I
#   + ω₁ I ⊗ n ⊗ I ⊗ I + ω₂ I ⊗ I ⊗ n ⊗ I + ω₃ I ⊗ I ⊗ I ⊗ n
#   + g σz ⊗ x ⊗ I ⊗ I + g σz ⊗ I ⊗ x ⊗ I + g σz ⊗ I ⊗ I ⊗ x

H_tt =
    # Spin terms
    0.5 * Δ0 * tt_mkron(σz, idb, idb, idb) +
    0.5 * Δx * tt_mkron(σx, idb, idb, idb) +
    # Boson number operators
    ω[1] * tt_mkron(id2, n_op, idb, idb) +
    ω[2] * tt_mkron(id2, idb, n_op, idb) +
    ω[3] * tt_mkron(id2, idb, idb, n_op) +
    # Spin-boson couplings
    g * tt_mkron(σz, x_op, idb, idb) +
    g * tt_mkron(σz, idb, x_op, idb) +
    g * tt_mkron(σz, idb, idb, x_op)

# Full Hamiltonian for exact solution (warning: scales as nb^3)
H_full =
    # Spin terms
    0.5 * Δ0 * kron(σz, idb, idb, idb) +
    0.5 * Δx * kron(σx, idb, idb, idb) +
    # Boson number operators
    ω[1] * kron(id2, n_op, idb, idb) +
    ω[2] * kron(id2, idb, n_op, idb) +
    ω[3] * kron(id2, idb, idb, n_op) +
    # Spin-boson couplings
    g * kron(σz, x_op, idb, idb) +
    g * kron(σz, idb, x_op, idb) +
    g * kron(σz, idb, idb, x_op)

opts = Dict(
    :verb => 1,
    :nswp => 2,
    :local_iters => 5,
    :time_scheme => "cn",
    :time_error_damp => 100.0,
    :kickrank => 4,
    :correct_2_norm => true,
)

# Total boson Hilbert space dimension
nb_total = nb^n_bosons
println("Total Hilbert space dimension: $(2 * nb_total)")

# Initial state: |↑, 0, 0, 0⟩ (spin up, all bosons in ground state)
x0 = initial_tt_state(spaces, state_index=1, site=1)
U = tkron(x0, tt_ones(nt; T=ComplexF64))

A = (-1im) * H_tt

ts = collect(range(0.0, Tfin; length=Nsteps + 1))

# Full initial state: |↑⟩ ⊗ |0⟩ ⊗ |0⟩ ⊗ |0⟩
boson_ground = ComplexF64[1.0; zeros(ComplexF64, nb - 1)]
psi0 = kron(ComplexF64[1.0, 0.0], boson_ground, boson_ground, boson_ground)

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

# Compute populations (nb_total = nb^n_bosons for multi-boson system)
p_up_exact, p_dn_exact = compute_spin_populations(psi_exact, nb_total)
p_up_tt, p_dn_tt = compute_spin_populations(psi_tt, nb_total)

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
