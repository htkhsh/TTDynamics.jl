using LinearAlgebra
using DelimitedFiles
using TTSolver
using TTDynamics
using QFiND
using KaisouEOM
using KaisouEOM: icm2ifs
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
println("QFiND bath modes: ", length(omega_bath))
println("QFiND relative BCF error: ", qfind_error)

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
println("ESPRIT exponential terms: ", length(fit.expon))
println("ESPRIT relative BCF error: ", fit_error)

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
println("Bosonic basis sizes: ", basis_sizes)
println("Initial TT ranks: ", target_ranks)
println("Total Hilbert-space dimension: ", 2 * prod(basis_sizes))
println("Initial state norm: ", real(dot(u, u)))

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

norm_figure = Figure(size=(900, 600))
norm_axis = Axis(
    norm_figure[1, 1];
    xlabel="Time [fs]",
    ylabel="Norm",
    title="KSL Norm Conservation",
)
lines!(norm_axis, times_ksl, norms_ksl; label="KSL norm")
hlines!(norm_axis, [1.0]; label="exact norm", linestyle=:dash, color=:black)
axislegend(norm_axis; position=:rt)
save("sb_ohmic_norm.png", norm_figure)

rank_figure = Figure(size=(900, 600))
rank_axis = Axis(
    rank_figure[1, 1];
    xlabel="Time [fs]",
    ylabel="TT rank",
    title="KSL TT Rank History",
)
for bond_index in 2:(length(rank_history[1]) - 1)
    lines!(
        rank_axis,
        times_ksl,
        getindex.(rank_history, bond_index);
        label="bond $(bond_index - 1)",
    )
end
lines!(
    rank_axis,
    times_ksl,
    maximum.(rank_history);
    label="maximum rank",
    color=:black,
    linewidth=3,
)
axislegend(rank_axis; position=:rt)
save("sb_ohmic_rank.png", rank_figure)
