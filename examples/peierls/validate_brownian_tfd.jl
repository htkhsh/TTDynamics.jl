using LinearAlgebra
using CairoMakie
using QFiND
using HEOMKit: icm2ifs
using TTDynamics
using TTSolver

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "utils.jl"))

function discretize_brownian_bath(; frequency_cm::Real, damping_cm::Real,
                                  reorganization_energy_cm::Real,
                                  temperature_K::Real, omega_min_cm::Real,
                                  omega_max_cm::Real, n_frequency::Integer,
                                  identification_time_fs::Real, n_time::Integer,
                                  identification_tolerance::Real)
    spectral_density = BrownianSD(frequency_cm, damping_cm,
                                  reorganization_energy_cm)
    qnsd = BosonicQNSD(spectral_density, temperature_K)
    β = QFiND.ħ * 1e15 / (QFiND.kb * temperature_K)
    # Brownian J(ω) is linear at the origin, so J(ω)coth(βω/2)
    # has a finite limit. QFiND's BosonicQNSD evaluates 0 * Inf at ω=0;
    # provide that analytic limit when an ID grid happens to contain zero.
    quantum_spectral_density = function (ω; scale=icm2ifs)
        if iszero(ω)
            return 2 * damping_cm * reorganization_energy_cm /
                   frequency_cm^2 * scale / (β * icm2ifs)
        end
        return qnsd(ω; scale)
    end
    # QFiND's direct Brownian BCF integrand has a removable singularity at
    # zero frequency; start just above zero to avoid evaluating 0/0 there.
    correlation = BosonicBCF(
        spectral_density, temperature_K, 1e-8,
        max(10_000.0, 10 * omega_max_cm), nothing, nothing,
    )
    dataset, estimated_bcf_error = InitialData(
        DiscrID(), quantum_spectral_density, correlation,
        omega_min_cm, omega_max_cm, identification_time_fs;
        n_freq=n_frequency, n_time=n_time,
    )
    identified = id_discr(dataset, identification_tolerance)
    frequencies_cm = Float64.(identified.freq)
    coefficients_cm = Float64.(identified.coeff)
    length(frequencies_cm) == length(coefficients_cm) ||
        throw(DimensionMismatch("QFiND returned different mode and coefficient counts"))
    isempty(frequencies_cm) && error("QFiND Brownian identification returned no modes")
    all(isfinite, frequencies_cm) && all(isfinite, coefficients_cm) ||
        error("QFiND Brownian identification returned nonfinite parameters")
    reconstructed = bcf_approx.(dataset.time, Ref(frequencies_cm),
                                Ref(coefficients_cm))
    relative_bcf_error = norm(reconstructed - dataset.bcf) / norm(dataset.bcf)
    return (; spectral_density, dataset, estimated_bcf_error,
            frequencies_cm, coefficients_cm, relative_bcf_error)
end

function brownian_heom_decomposition(config; frequency_cm::Real,
                                     damping_cm::Real,
                                     reorganization_energy_cm::Real)
    spectral_density = BrownianSD(frequency_cm, damping_cm,
                                  reorganization_energy_cm)
    exponents, coefficients = tpsd(
        spectral_density, config.temperature_K, config.pade_order,
        config.tpsd_tolerance; pade_type=config.pade_type,
    )
    exponents = ComplexF64.(exponents)
    coefficients = ComplexF64.(coefficients)
    isempty(exponents) && error("Brownian TPSD returned no exponential terms")
    all(isfinite, exponents) && all(>(0), real.(exponents)) ||
        error("Brownian TPSD returned invalid exponents")
    all(isfinite, coefficients) ||
        error("Brownian TPSD returned nonfinite coefficients")
    return (; exponents, coefficients)
end

function _gibbs_density(hamiltonian, temperature_K)
    temperature_K > 0 || throw(ArgumentError("temperature_K must be positive"))
    eigensystem = eigen(Hermitian(hamiltonian))
    β = QFiND.ħ * 1e15 / (QFiND.kb * temperature_K)
    weights = exp.(-β .* (eigensystem.values .- minimum(eigensystem.values)))
    weights ./= sum(weights)
    return eigensystem.vectors * Diagonal(weights) * eigensystem.vectors'
end

function build_brownian_tfd_model(config, bath; boson_size::Integer=2)
    n = config.site_count
    n >= 3 || throw(ArgumentError("Peierls validation requires at least three sites"))
    boson_size >= 2 || throw(ArgumentError("boson_size must be at least two"))
    length(bath.frequencies_cm) == length(bath.coefficients_cm) ||
        throw(DimensionMismatch("bath frequency and coefficient counts differ"))
    system_hamiltonian = peierls_hamiltonian(config) * icm2ifs
    ancilla_identity = Matrix{ComplexF64}(I, n, n)
    identity_system = Matrix{ComplexF64}(I, n^2, n^2)
    couplings = peierls_couplings(n)
    frequencies_fs = bath.frequencies_cm .* icm2ifs
    coefficients_fs = bath.coefficients_cm .* icm2ifs
    baths = [BosonicTFD(frequencies_fs, coefficients_fs,
                        kron(ancilla_identity, coupling))
             for coupling in couplings]
    tfd_system = SBSystem(kron(ancilla_identity, system_hamiltonian),
                          BosonicEnv(baths))
    mode_count = n * length(frequencies_fs)
    basis_sizes = fill(Int(boson_size), mode_count)
    hamiltonian, _ = tt_sbham(tfd_system; basis_sizes,
                              unit_conv=1.0)

    local_identities = [Matrix{ComplexF64}(I, size, size) for size in basis_sizes]
    coordinates = [Matrix(BosonSpace(size).x) for size in basis_sizes]
    bond_data = bond_operators(n)
    local_current = [-im * difference for difference in bond_data.antisymmetric]
    current = nothing
    hopping = config.hopping_cm * icm2ifs
    modes_per_site = length(frequencies_fs)
    for site in 1:n
        electronic_current = kron(ancilla_identity, local_current[site])
        factors = copy(local_identities)
        term = (-hopping) * tt_mkron(electronic_current, factors...)
        next_site = mod1(site + 1, n)
        for mode_local in 1:modes_per_site
            mode = (site - 1) * modes_per_site + mode_local
            factors[mode] = coordinates[mode]
            term += coefficients_fs[mode_local] *
                    tt_mkron(electronic_current, factors...)
            factors[mode] = local_identities[mode]

            next_mode = (next_site - 1) * modes_per_site + mode_local
            factors[next_mode] = coordinates[next_mode]
            term -= coefficients_fs[mode_local] *
                    tt_mkron(electronic_current, factors...)
            factors[next_mode] = local_identities[next_mode]
        end
        term = tt_round(term, config.operator_tolerance)
        current = isnothing(current) ? term :
                  tt_round(current + term, config.operator_tolerance)
    end

    density = _gibbs_density(system_hamiltonian, config.temperature_K)
    eigen_density = eigen(Hermitian(density))
    purification = vec(eigen_density.vectors *
                       Diagonal(sqrt.(max.(eigen_density.values, 0))) *
                       eigen_density.vectors')
    cores = [reshape(ComplexF64.(purification), 1, n^2, 1)]
    for size in basis_sizes
        push!(cores, reshape(ComplexF64[1; zeros(size - 1)], 1, size, 1))
    end
    initial_state = TTTensor(cores)
    return (; hamiltonian, current, initial_state, basis_sizes, mode_count)
end

function _tfd_tamen_step(state::TTTensor, generator::TTMatrix, config, workspace;
                         tolerance::Real=get(config, :tfd_tamen_tolerance,
                                             config.correlation_tamen_tolerance))
    extended = tt_kron(state, tt_ones(config.temporal_basis_size; T=ComplexF64))
    solution, time_grid, _, _, _ = tamen(
        extended, config.time_step_fs * generator,
        tolerance;
        verb=0, nswp=config.sweep_count,
        local_iters=config.local_iterations, kickrank=config.kick_rank,
        time_scheme="CN", time_error_damp=100.0, correct_2_norm=false,
        workspace,
    )
    return extract_snapshot(solution, time_grid, 1.0, "CN")
end

function run_brownian_tfd_correlation(model, config; duration_fs::Real,
                                      tamen_tolerance::Real=get(
                                          config, :tfd_tamen_tolerance,
                                          config.correlation_tamen_tolerance),
                                      progress_callback=nothing)
    count = step_count(duration_fs, config.time_step_fs; allow_zero=true)
    state = model.initial_state
    source = tt_round(model.current * state, config.state_rounding_tolerance)
    generator = -im * model.hamiltonian
    state_workspace = TAMEnWorkspace(ComplexF64)
    source_workspace = TAMEnWorkspace(ComplexF64)
    correlation = ComplexF64[tt_dot(state, model.current * source)]
    maximum_rank = Int[max(maximum(tt_ranks(state)), maximum(tt_ranks(source)))]
    progress_callback === nothing || progress_callback((;
        step=0, time_fs=0.0, correlation=first(correlation),
        maximum_rank=first(maximum_rank)))
    for step in 1:count
        state = _tfd_tamen_step(state, generator, config, state_workspace;
                                tolerance=tamen_tolerance)
        source = _tfd_tamen_step(source, generator, config, source_workspace;
                                 tolerance=tamen_tolerance)
        push!(correlation, tt_dot(state, model.current * source))
        push!(maximum_rank, max(maximum(tt_ranks(state)), maximum(tt_ranks(source))))
        if progress_callback !== nothing &&
           (step == count || step % config.progress_interval == 0)
            progress_callback((; step, time_fs=step * config.time_step_fs,
                               correlation=correlation[end],
                               maximum_rank=maximum_rank[end]))
        end
    end
    return (; times=collect(0:count) .* config.time_step_fs,
            correlation, maximum_rank, state, source)
end

function brownian_validation_config(; temperature_K::Real=300.0,
                                    hierarchy_local_size::Integer=20,
                                    reorganization_energy_cm::Real=50.0,
                                    heom_tamen_tolerance::Real=1e-2,
                                    tfd_tamen_tolerance::Real=1e-2)
    hierarchy_local_size >= 1 ||
        throw(ArgumentError("hierarchy_local_size must be positive"))
    all(isfinite, (heom_tamen_tolerance, tfd_tamen_tolerance)) &&
        all(>(0), (heom_tamen_tolerance, tfd_tamen_tolerance)) ||
        throw(ArgumentError("HEOM and TFD TAMEn tolerances must be finite and positive"))
    return merge(DEFAULT_CONFIG, (; site_count=3,
        site_energies_cm=zeros(Float64, 3), hopping_cm=50.0,
        reorganization_energy_cm=Float64(reorganization_energy_cm),
        temperature_K=Float64(temperature_K), initial_condition=:gibbs,
        hierarchy_local_size=Int(hierarchy_local_size), pade_order=3,
        tpsd_tolerance=2e-2, time_step_fs=1.0,
        correlation_tamen_tolerance=1e-2,
        heom_tamen_tolerance=Float64(heom_tamen_tolerance),
        tfd_tamen_tolerance=Float64(tfd_tamen_tolerance),
        sweep_count=2, local_iterations=3, kick_rank=2,
        progress_interval=10))
end

function run_brownian_tfd_validation(; duration_fs::Real=100.0,
                                     frequency_cm::Real=200.0,
                                     damping_cm::Real=50.0,
                                     reorganization_energy_cm::Real=50.0,
                                     temperature_K::Real=100.0,
                                     identification_tolerance::Real=1e-3,
                                     hierarchy_local_size::Integer=20,
                                     boson_size::Integer=50,
                                     heom_tamen_tolerance::Real=2e-3,
                                     tfd_tamen_tolerance::Real=2e-3)
    config = brownian_validation_config(; temperature_K,
        hierarchy_local_size, reorganization_energy_cm,
        heom_tamen_tolerance, tfd_tamen_tolerance)
    configure_blas(config)
    bath = discretize_brownian_bath(
        frequency_cm=frequency_cm, damping_cm=damping_cm,
        reorganization_energy_cm=reorganization_energy_cm,
        temperature_K=temperature_K, omega_min_cm=-1000.0,
        omega_max_cm=1500.0, n_frequency=2000,
        identification_time_fs=duration_fs, n_time=200,
        identification_tolerance=identification_tolerance,
    )
    decomposition = brownian_heom_decomposition(
        config; frequency_cm, damping_cm, reorganization_energy_cm)
    heom_problem = build_problem(config, decomposition)
    equilibrium = initial_heom_state(config, heom_problem.system)
    progress(label) = item -> println(
        "  $(label) step=$(item.step) t=$(item.time_fs) fs " *
        "CCF=$(item.correlation) rmax=$(item.maximum_rank)")
    heom = run_current_correlation(
        equilibrium, config, heom_problem; duration_fs,
        tamen_tolerance=config.heom_tamen_tolerance,
        progress_callback=progress("HEOM-TAMEn"),
    )
    tfd_model = build_brownian_tfd_model(config, bath; boson_size)
    tfd = run_brownian_tfd_correlation(
        tfd_model, config; duration_fs,
        tamen_tolerance=config.tfd_tamen_tolerance,
        progress_callback=progress("TFD-tAMEn"),
    )
    length(heom.times) == length(tfd.times) &&
        all(isapprox.(heom.times, tfd.times; atol=1e-12, rtol=0)) ||
        throw(DimensionMismatch("HEOM and TFD time grids differ"))
    return (; config, bath, decomposition, heom, tfd, tfd_model,
            maximum_difference=maximum(abs.(heom.correlation - tfd.correlation)))
end

function save_brownian_tfd_plot(path::AbstractString, result)
    figure = Figure(size=(980, 720))
    title = "Peierls CCF: Brownian HEOM vs TFD (tAMEn)"
    real_axis = Axis(figure[1, 1]; xlabel="Time (fs)", ylabel="Re C(t) (fs⁻²)",
                     title)
    lines!(real_axis, result.heom.times, real.(result.heom.correlation);
           linewidth=2, label="HEOM-TAMEn")
    lines!(real_axis, result.tfd.times, real.(result.tfd.correlation);
           linewidth=2, linestyle=:dash, label="TFD-tAMEn")
    axislegend(real_axis; position=:rt)
    imaginary_axis = Axis(figure[2, 1]; xlabel="Time (fs)",
                          ylabel="Im C(t) (fs⁻²)")
    lines!(imaginary_axis, result.heom.times, imag.(result.heom.correlation);
           linewidth=2, label="HEOM-TAMEn")
    lines!(imaginary_axis, result.tfd.times, imag.(result.tfd.correlation);
           linewidth=2, linestyle=:dash, label="TFD-tAMEn")
    axislegend(imaginary_axis; position=:rb)
    linkxaxes!(real_axis, imaginary_axis)
    save(String(path), figure)
    return String(path)
end

function brownian_tfd_main(config=DEFAULT_CONFIG)
    println("Peierls Brownian bath: HEOM-TAMEn vs TFD-tAMEn")
    result = run_brownian_tfd_validation()
    println("  Brownian modes from QFiND ID: $(length(result.bath.frequencies_cm))")
    println("  QFiND relative BCF fit error: $(result.bath.relative_bcf_error)")
    println("  Brownian HEOM exponential terms: $(length(result.decomposition.exponents))")
    println("  TFD local boson dimensions: $(result.tfd_model.basis_sizes)")
    println("  max |HEOM-TAMEn - TFD-tAMEn| = $(result.maximum_difference)")
    configured_directory = String(config.output_directory)
    output_directory = isabspath(configured_directory) ?
        configured_directory : joinpath(@__DIR__, configured_directory)
    mkpath(output_directory)
    output_path = joinpath(output_directory, "peierls_brownian_tfd.png")
    save_brownian_tfd_plot(output_path, result)
    println("  wrote $output_path")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    brownian_tfd_main()
end
