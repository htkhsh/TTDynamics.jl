using LinearAlgebra
using QuadGK
using TTDynamics
using Test
using TTSolver

include("../config.jl")
include("../oscillator.jl")
include("../correlation.jl")
include("../model.jl")
include("../dynamics.jl")

function quartic_test_config_with(config::QuarticConfig; kwargs...)
    names = fieldnames(QuarticConfig)
    values = NamedTuple{names}(Tuple(getfield(config, name) for name in names))
    return QuarticConfig(; merge(values, kwargs)...)
end

function quartic_test_fit()
    metadata = CorrelationFitMetadata(
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        (0.0, 1.0),
        [1.0],
        1.0,
        Inf,
        0.2,
        String[],
    )
    return ExponentialCorrelation(
        ComplexF64[0.7 + 0.1im],
        ComplexF64[0.16 - 0.04im],
        ComplexF64[0.16 + 0.04im],
        metadata,
    )
end

function quartic_test_projector(dimension, index)
    operator = zeros(ComplexF64, dimension, dimension)
    operator[index, index] = 1
    return operator
end

function quartic_test_liouvillian_term(factors)
    return -1im * (
        tt_mkron(liouville_left.(factors)) -
        tt_mkron(liouville_right.(factors))
    )
end

function quartic_test_product_vector(dimensions)
    local_vectors = [ComplexF64.(1:dimension) ./ dimension for dimension in dimensions]
    return TTTensor([
        reshape(vector, 1, length(vector), 1)
        for vector in local_vectors
    ])
end

function quartic_test_zero_result(model, config)
    populations = zeros(Float64, config.site_count, 1)
    populations[1, 1] = 1
    return (;
        times=[0.0],
        populations,
        trace=ComplexF64[1],
        electron_number=ComplexF64[1],
        oscillator_q=zeros(config.site_count, 1),
        oscillator_q2=zeros(config.site_count, 1),
        oscillator_energy=zeros(config.site_count, 1),
        hermiticity_error=[0.0],
        maximum_rank=[1],
        mean_rank=[1.0],
        tamen_residual=Union{Missing,Float64}[missing],
        tamen_truncation=Union{Missing,Float64}[missing],
        final_state=model.initial_state,
    )
end

@testset "quartic README distinguishes short-time and plumbing commands" begin
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)
    required_command = """julia --project=examples/quartic -e 'include(\"examples/quartic/two_site_transfer.jl\"); two_site_main(QuarticConfig(final_time=0.1, time_step=0.1, d_raw=6, d_keep=2))'"""

    @test occursin(required_command, readme)
    @test occursin("potentially expensive short-time two-site invocation", readme)
    @test occursin("actual plumbing smoke", readme)
    @test !occursin("For a short two-site smoke run", readme)
end

@testset "quartic convergence dispatcher rebuilds every requested case" begin
    base_config = QuarticConfig(
        site_count=2,
        site_energies=[0.1, -0.1],
        d_raw=8,
        d_keep=2,
        basis_frequency=0.75,
        hierarchy_nmax=1,
        fit_rank=2,
        fit_rank_max=6,
        final_time=0.2,
        time_step=0.1,
        operator_tolerance=1e-8,
        state_rounding_tolerance=1e-8,
    )
    original_fields = NamedTuple{fieldnames(QuarticConfig)}(
        Tuple(deepcopy(getfield(base_config, name)) for name in fieldnames(QuarticConfig)),
    )
    sweeps = (
        d_raw=[8, 10],
        d_keep=[2, 3],
        basis_frequency=[0.6, 0.9],
        hierarchy_nmax=[0, 2],
        fit_rank=[1, 3],
        tt_cutoff=[1e-7, 1e-9],
        time_step=[0.2, 0.1],
    )

    for parameter in keys(sweeps)
        values = getfield(sweeps, parameter)
        calls = QuarticConfig[]
        runner = config -> begin
            push!(calls, config)
            run_index = length(calls)
            return (;
                populations=[1.0 1.0 - 0.1run_index; 0.0 0.1run_index],
                trace=ComplexF64[1.0, 1.0 - run_index * 1e-10],
                fit_error=run_index * 1e-4,
                maximum_rank=[1, run_index + 2],
            )
        end

        diagnostics = run_quartic_convergence(
            base_config;
            parameter,
            values,
            runner,
        )

        @test length(diagnostics) == 2
        @test getfield.(diagnostics, :label) ==
              ["$(parameter)=$(repr(value))" for value in values]
        @test getfield.(diagnostics, :parameter) == fill(parameter, 2)
        @test getfield.(diagnostics, :value) == collect(values)
        @test diagnostics[1].final_populations == [0.9, 0.1]
        @test diagnostics[2].final_populations == [0.8, 0.2]
        @test diagnostics[2].trace == 1.0 - 2e-10
        @test diagnostics[2].fit_error == 2e-4
        @test diagnostics[2].maximum_rank == 4
        @test length(calls) == 2
        @test all(config -> config !== base_config, calls)
        @test calls[1] !== calls[2]

        if parameter === :tt_cutoff
            @test getfield.(calls, :operator_tolerance) == values
            @test getfield.(calls, :state_rounding_tolerance) == values
        else
            @test getfield.(calls, parameter) == values
        end
    end

    @test NamedTuple{fieldnames(QuarticConfig)}(
        Tuple(getfield(base_config, name) for name in fieldnames(QuarticConfig)),
    ) == original_fields
    @test_throws ArgumentError run_quartic_convergence(
        base_config;
        parameter=:temperature,
        values=[0.5],
        runner=identity,
    )
end

@testset "quartic bath correlation sampling" begin
    cfg = QuarticConfig(
        temperature=0.5,
        bath_lambda=0.7,
        bath_gamma=1.3,
        quadrature_atol=1e-10,
        quadrature_rtol=1e-8,
    )
    bath = CriticallyDampedBrownian(cfg.bath_lambda, cfg.bath_gamma)
    normalization, _ = quadgk(
        ω -> spectral_density(bath, ω) / ω,
        0.0,
        Inf;
        rtol=1e-10,
    )
    @test normalization / π ≈ bath.lambda rtol=1e-8
    @test isfinite(bath_correlation(bath, 0.0, cfg.temperature; rtol=1e-8, atol=1e-10))

    times = collect(0.0:0.1:0.5)
    samples = sample_bath_correlation(
        bath,
        cfg.temperature,
        times;
        rtol=cfg.quadrature_rtol,
        atol=cfg.quadrature_atol,
    )
    @test length(samples) == length(times)
    @test all(isfinite, samples)

    cutoff_20 = sample_bath_correlation(
        bath,
        cfg.temperature,
        times;
        omega_integration_max=20 * bath.gamma,
        rtol=1e-9,
        atol=1e-11,
    )
    cutoff_40 = sample_bath_correlation(
        bath,
        cfg.temperature,
        times;
        omega_integration_max=40 * bath.gamma,
        rtol=1e-9,
        atol=1e-11,
    )
    @test cutoff_20 ≈ cutoff_40 rtol=2e-3

    looser = sample_bath_correlation(
        bath,
        cfg.temperature,
        times;
        omega_integration_max=40 * bath.gamma,
        rtol=1e-7,
        atol=1e-9,
    )
    tighter = sample_bath_correlation(
        bath,
        cfg.temperature,
        times;
        omega_integration_max=40 * bath.gamma,
        rtol=1e-9,
        atol=1e-11,
    )
    @test looser ≈ tighter rtol=5e-4
end

@testset "quartic bath correlation rejects invalid quadrature controls" begin
    bath = CriticallyDampedBrownian(0.7, 1.3)

    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; rtol=Inf)
    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; rtol=NaN)
    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; atol=Inf)
    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; atol=NaN)
    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; maxevals=0)
    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; maxevals=-1)
    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; maxevals=1.5)
end

@testset "complex ESPRIT uses a conjugate-closed common pole basis" begin
    times = collect(0.0:0.05:4.0)
    rates = ComplexF64[0.3 + 0.7im, 0.3 - 0.7im, 0.305 + 0.705im]
    coefficients = ComplexF64[1.0 + 0.2im, 0.4 - 0.1im, 0.03 + 0.02im]
    samples = ComplexF64[
        sum(coefficients[index] * exp(-rates[index] * time) for index in eachindex(rates))
        for time in times
    ]
    validation_times = collect(0.025:0.05:3.975)
    validation_samples = ComplexF64[
        sum(coefficients[index] * exp(-rates[index] * time) for index in eachindex(rates))
        for time in validation_times
    ]

    fit = fit_correlation_esprit(
        samples,
        times;
        fit_rank=3,
        pole_stability_tolerance=1e-10,
        duplicate_pole_tolerance=1e-8,
        fit_relative_tolerance=1e-8,
        validation_samples=validation_samples,
        validation_times=validation_times,
    )

    @test norm(evaluate_correlation(fit, times) - samples) / norm(samples) < 1e-8
    @test norm(evaluate_correlation(fit, times; branch=:backward) - conj.(samples)) /
          norm(samples) < 1e-8
    @test length(fit.rates) == 4
    @test 0 < fit.metadata.minimum_pole_separation < 0.01
    @test all(
        rate -> any(
            other -> isapprox(rate, conj(other); atol=1e-8, rtol=0),
            fit.rates,
        ),
        fit.rates,
    )
end

@testset "complex ESPRIT requires disjoint validation data" begin
    training_times = collect(0.0:0.1:2.0)
    samples = ComplexF64[exp(-(0.4 + 0.2im) * time) for time in training_times]

    @test_throws ArgumentError fit_correlation_esprit(
        samples,
        training_times;
        fit_rank=1,
    )
    @test_throws ArgumentError fit_correlation_esprit(
        samples,
        training_times;
        fit_rank=1,
        validation_samples=samples,
        validation_times=training_times,
    )
end

@testset "complex ESPRIT rejects invalid grids and unstable poles" begin
    uniform_times = collect(0.0:0.1:2.0)
    decaying_samples = ComplexF64[exp(-(0.4 + 0.2im) * time) for time in uniform_times]
    validation_times = collect(0.05:0.1:1.95)
    decaying_validation =
        ComplexF64[exp(-(0.4 + 0.2im) * time) for time in validation_times]
    nonuniform_times = copy(uniform_times)
    nonuniform_times[8] += 0.01
    @test_throws ArgumentError fit_correlation_esprit(
        decaying_samples,
        nonuniform_times;
        fit_rank=1,
        validation_samples=decaying_validation,
        validation_times=validation_times,
    )

    growing_samples = ComplexF64[exp(0.2 * time) for time in uniform_times]
    growing_validation = ComplexF64[exp(0.2 * time) for time in validation_times]
    @test_throws ArgumentError fit_correlation_esprit(
        growing_samples,
        uniform_times;
        fit_rank=1,
        validation_samples=growing_validation,
        validation_times=validation_times,
    )

    dt = uniform_times[2] - uniform_times[1]
    @test_throws ArgumentError _validated_conjugate_closed_rates(
        ComplexF64[0.2 + (π / dt + 1e-6) * im],
        dt;
        pole_stability_tolerance=1e-10,
        duplicate_pole_tolerance=1e-8,
    )
    corrected_rates, corrections = _validated_conjugate_closed_rates(
        ComplexF64[-5e-11 + 0.3im],
        dt;
        pole_stability_tolerance=1e-10,
        duplicate_pole_tolerance=1e-8,
    )
    @test real(first(corrected_rates)) == 0
    @test any(contains("corrected to zero"), corrections)

    @test_throws ArgumentError _validated_conjugate_closed_rates(
        ComplexF64[-5e-11 + 0im, 0.0 + 0im],
        dt;
        pole_stability_tolerance=1e-10,
        duplicate_pole_tolerance=1e-8,
    )
end

@testset "correlation validation rejects either inaccurate branch" begin
    rates = ComplexF64[0.4 + 0.2im, 0.4 - 0.2im]
    coeff_forward = ComplexF64[0.7 + 0.1im, 0.2 - 0.05im]
    coeff_backward = zeros(ComplexF64, 2)
    metadata = CorrelationFitMetadata(
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        (0.0, 1.0),
        [1.0, 0.5],
        2.0,
        0.4,
        maximum(abs, coeff_forward),
        String[],
    )
    fit = ExponentialCorrelation(rates, coeff_forward, coeff_backward, metadata)
    validation_times = collect(0.05:0.1:0.95)
    validation_samples = ComplexF64[
        sum(
            coeff_forward[index] * exp(-rates[index] * time)
            for index in eachindex(rates)
        ) for time in validation_times
    ]

    @test_throws ArgumentError validate_correlation_fit(
        fit,
        validation_samples,
        validation_times;
        fit_absolute_tolerance=1e-10,
        fit_relative_tolerance=1e-10,
    )
end

@testset "complex ESPRIT validates cBO fits away from training points" begin
    bath = CriticallyDampedBrownian(0.2, 1.1)
    temperature = 0.7
    training_times = collect(0.0:0.2:2.0)
    validation_times = collect(0.1:0.2:1.9)
    training_samples = sample_bath_correlation(
        bath,
        temperature,
        training_times;
        omega_integration_max=40 * bath.gamma,
        rtol=1e-8,
        atol=1e-10,
    )
    validation_samples = sample_bath_correlation(
        bath,
        temperature,
        validation_times;
        omega_integration_max=40 * bath.gamma,
        rtol=1e-8,
        atol=1e-10,
    )

    fit = fit_correlation_esprit(
        training_samples,
        training_times;
        fit_rank=4,
        fit_absolute_tolerance=1e-4,
        fit_relative_tolerance=0.1,
        validation_samples=validation_samples,
        validation_times=validation_times,
    )
    holdout_forward_error =
        norm(evaluate_correlation(fit, validation_times) - validation_samples) /
        norm(validation_samples)
    holdout_backward_error = norm(
        evaluate_correlation(fit, validation_times; branch=:backward) -
        conj.(validation_samples),
    ) / norm(validation_samples)
    training_forward_error =
        norm(evaluate_correlation(fit, training_times) - training_samples) /
        norm(training_samples)
    training_backward_error = norm(
        evaluate_correlation(fit, training_times; branch=:backward) -
        conj.(training_samples),
    ) / norm(training_samples)
    @test max(holdout_forward_error, holdout_backward_error) < 0.1
    @test fit.metadata.validation_relative_error ≈
          max(holdout_forward_error, holdout_backward_error) rtol=1e-10
    @test fit.metadata.training_relative_error ≈
          max(training_forward_error, training_backward_error) rtol=1e-10
    @test all(isfinite, fit.metadata.singular_values)
    @test all(
        isfinite,
        (
            fit.metadata.training_absolute_error,
            fit.metadata.training_relative_error,
            fit.metadata.validation_absolute_error,
            fit.metadata.validation_relative_error,
            fit.metadata.sampling_interval,
            fit.metadata.vandermonde_condition,
            fit.metadata.minimum_pole_separation,
            fit.metadata.maximum_coefficient_magnitude,
        ),
    )

    rank_errors = Float64[]
    for rank in (3, 4)
        rank_fit = fit_correlation_esprit(
            training_samples,
            training_times;
            fit_rank=rank,
            fit_absolute_tolerance=1e-4,
            fit_relative_tolerance=0.5,
            validation_samples=validation_samples,
            validation_times=validation_times,
        )
        diagnostics = validate_correlation_fit(
            rank_fit,
            validation_samples,
            validation_times;
            fit_absolute_tolerance=1e-4,
            fit_relative_tolerance=0.5,
        )
        push!(rank_errors, diagnostics.relative_error)
    end
    @test length(rank_errors) == 2
    @test all(isfinite, rank_errors)
    @test all(<(0.5), rank_errors)
end

@testset "quartic physical Hamiltonian has local Hilbert-core ordering and no counter term" begin
    config = QuarticConfig(
        site_count=2,
        site_energies=[0.25, -0.15],
        hopping=0.4,
        g=0.3,
        d_raw=8,
        d_keep=3,
        hierarchy_nmax=1,
        operator_tolerance=1e-14,
    )
    mode = build_quartic_mode(config)
    fit = quartic_test_fit()
    model = build_quartic_model(config, mode, fit; initial_site=1)

    electron_hamiltonian = ComplexF64[0.25 -0.4; -0.4 -0.15]
    I2 = Matrix{ComplexF64}(I, 2, 2)
    I3 = Matrix{ComplexF64}(I, 3, 3)
    expected = kron(electron_hamiltonian, I3, I3) +
               kron(I2, mode.hamiltonian, I3) +
               kron(I2, I3, mode.hamiltonian) +
               config.g * kron(quartic_test_projector(2, 1), mode.q, I3) +
               config.g * kron(quartic_test_projector(2, 2), I3, mode.q)

    @test tt_full(model.system_hamiltonian) ≈ expected atol=1e-11
    @test model.system.physical_dimensions == [4, 9, 9]
    @test getfield.(model.system.couplings, :physical_core) == [2, 3]
    @test all(coupling -> coupling.correlation === fit, model.system.couplings)
    @test model.system.hierarchy_sizes == [2, 2]

    changed_bath = quartic_test_config_with(
        config;
        bath_lambda=2config.bath_lambda,
        bath_gamma=3config.bath_gamma,
        temperature=2config.temperature,
    )
    @test tt_full(build_system_mpo(changed_bath, mode)) ≈
          tt_full(build_system_mpo(config, mode)) atol=1e-11
end

@testset "quartic factorized state and physical observables" begin
    config = QuarticConfig(
        site_count=2,
        d_raw=8,
        d_keep=3,
        hierarchy_nmax=1,
        operator_tolerance=1e-12,
        state_rounding_tolerance=1e-12,
    )
    mode = build_quartic_mode(config)
    model = build_quartic_model(config, mode, quartic_test_fit(); initial_site=2)
    observables = model.observables
    state = model.initial_state

    populations = ComplexF64[
        root_expectation(state, model.system, observable)
        for observable in observables.electron_populations
    ]
    @test root_expectation(state, model.system, observables.trace) ≈ 1 atol=1e-11
    @test populations ≈ [0, 1] atol=1e-11
    @test sum(populations) ≈ 1 atol=1e-11
    @test root_expectation(state, model.system, observables.electron_number) ≈ 1 atol=1e-11
    @test length(observables.oscillator_q) == config.site_count
    @test length(observables.oscillator_q2) == config.site_count
    @test length(observables.oscillator_hamiltonian) == config.site_count
    @test all(
        observable -> tt_dims(observable) == model.system.physical_dimensions,
        [
            observables.electron_populations
            observables.oscillator_q
            observables.oscillator_q2
            observables.oscillator_hamiltonian
            [observables.trace, observables.electron_number]
        ],
    )
end

@testset "quartic zero coupling factorizes and zero bath takes the closed-system path" begin
    config = QuarticConfig(
        site_count=2,
        site_energies=[0.2, -0.1],
        hopping=0.35,
        g=0.0,
        d_raw=6,
        d_keep=2,
        hierarchy_nmax=1,
        operator_tolerance=1e-12,
    )
    mode = build_quartic_mode(config)
    fit = quartic_test_fit()
    model = build_quartic_model(config, mode, fit)

    electron_hamiltonian = ComplexF64[0.2 -0.35; -0.35 -0.1]
    I2 = Matrix{ComplexF64}(I, 2, 2)
    physical_liouvillian = quartic_test_liouvillian_term(
        [electron_hamiltonian, I2, I2],
    )
    for site in 1:config.site_count
        factors = [I2, I2, I2]
        factors[site + 1] = mode.hamiltonian
        physical_liouvillian += quartic_test_liouvillian_term(factors)
    end
    expected_system = MultiCoreHEOMTTSystem(
        physical_liouvillian,
        [4, 4, 4],
        [
            LocalBathCoupling(2, mode.q, fit),
            LocalBathCoupling(3, mode.q, fit),
        ],
        [2, 2],
    )
    expected_generator = build_multicore_heom_generator(
        expected_system;
        tol=config.operator_tolerance,
    )
    generator_probe = quartic_test_product_vector(multicore_heom_dimensions(model.system))
    generator_action = model.generator * generator_probe
    expected_action = expected_generator * generator_probe
    dense_generator_action = vec(tt_full(generator_action))
    dense_expected_action = vec(tt_full(expected_action))
    @test norm(dense_generator_action - dense_expected_action) <=
          1e-10 * max(norm(dense_expected_action), 1)

    no_bath_config = quartic_test_config_with(config; bath_lambda=0.0)
    no_bath = build_quartic_model(no_bath_config, mode, fit)
    @test isempty(no_bath.system.couplings)
    @test isempty(no_bath.system.hierarchy_sizes)
    @test multicore_heom_dimensions(no_bath.system) == [4, 4, 4]
    closed_probe = quartic_test_product_vector(multicore_heom_dimensions(no_bath.system))
    closed_action = no_bath.generator * closed_probe
    physical_action = no_bath.system.physical_liouvillian * closed_probe
    dense_closed_action = vec(tt_full(closed_action))
    dense_physical_action = vec(tt_full(physical_action))
    @test norm(dense_closed_action - dense_physical_action) <=
          1e-10 * max(norm(dense_physical_action), 1)
    @test build_quartic_model(no_bath_config, mode, nothing).fit === nothing
end

@testset "quartic root measurement and output safety" begin
    config = QuarticConfig(
        site_count=2,
        d_raw=6,
        d_keep=2,
        bath_lambda=0.0,
        final_time=0.1,
        time_step=0.1,
        operator_tolerance=1e-12,
        state_rounding_tolerance=1e-12,
    )
    mode = build_quartic_mode(config)
    model = build_quartic_model(config, mode, nothing; initial_site=1)

    measurement = measure_quartic_state(model.initial_state, model)
    @test measurement.trace ≈ 1 atol=1e-11
    @test measurement.electron_number ≈ 1 atol=1e-11
    @test measurement.hermiticity_error < 1e-11
    @test measurement.populations ≈ [1.0, 0.0] atol=1e-11
    @test all(isfinite, measurement.oscillator_q)
    @test all(isfinite, measurement.oscillator_q2)
    @test all(isfinite, measurement.oscillator_energy)

    harmonic_config = quartic_test_config_with(config; K4=0.0)
    harmonic_mode = build_quartic_mode(harmonic_config)
    harmonic_model = build_quartic_model(harmonic_config, harmonic_mode, nothing)
    harmonic_measurement = measure_quartic_state(
        harmonic_model.initial_state,
        harmonic_model,
    )
    @test harmonic_measurement.hermiticity_error < 1e-14

    mktempdir() do directory
        result = propagate_quartic_heom(model, config; final_time=0.0)
        path = joinpath(directory, "quartic.csv")
        @test write_quartic_csv(path, result) == path
        header = split(readline(path), ',')
        @test header[1:3] == ["time", "population_site_1", "population_site_2"]
        @test all(
            in(header),
            [
                "trace_real",
                "trace_imag",
                "hermiticity_error",
                "electron_number_real",
                "electron_number_imag",
                "oscillator_q_site_1",
                "oscillator_q2_site_1",
                "oscillator_energy_site_1",
                "maximum_rank",
                "mean_rank",
                "tamen_residual",
                "tamen_truncation",
            ],
        )
        @test_throws ArgumentError write_quartic_csv(path, result)
        @test write_quartic_csv(path, result; overwrite=true) == path

        paths = quartic_output_paths(directory; stem="smoke")
        @test paths.csv == joinpath(directory, "smoke.csv")
        @test paths.oscillator_q2 == joinpath(directory, "smoke_q2.png")
    end
end

@testset "quartic measurement rejects nonfinite complex observables" begin
    @test_throws ArgumentError _quartic_real_expectations(
        ComplexF64[complex(1.0, NaN), complex(1.0, Inf)],
        "test observable",
    )
end

@testset "quartic Crank--Nicolson propagation" begin
    closed_config = QuarticConfig(
        site_count=1,
        site_energies=[0.15],
        hopping=0.0,
        g=0.3,
        d_raw=6,
        d_keep=2,
        bath_lambda=0.0,
        final_time=0.01,
        time_step=0.01,
        temporal_basis_size=3,
        tamen_tolerance=1e-5,
        operator_tolerance=1e-12,
        state_rounding_tolerance=1e-12,
        sweep_count=4,
        local_iterations=8,
        kick_rank=2,
    )
    closed_mode = build_quartic_mode(closed_config)
    closed_model = build_quartic_model(closed_config, closed_mode, nothing)
    closed_result = propagate_quartic_heom(closed_model, closed_config)

    initial_density = kron(
        ComplexF64[1;;],
        quartic_thermal_density(closed_mode, closed_config.temperature),
    )
    dense_hamiltonian = tt_full(closed_model.system_hamiltonian)
    propagator = exp(-1im * dense_hamiltonian * closed_config.time_step)
    expected_density = propagator * initial_density * propagator'
    propagated_root = root_ado(closed_result.final_state, closed_model.system)
    propagated_density = reshape(vec(tt_full(propagated_root)), 2, 2)

    @test propagated_density ≈ expected_density atol=2e-5
    @test closed_result.trace[end] ≈ 1 atol=2e-6
    @test closed_result.electron_number[end] ≈ 1 atol=2e-6
    @test closed_result.hermiticity_error[end] < 2e-6

    bath_config = quartic_test_config_with(
        closed_config;
        bath_lambda=0.1,
        final_time=0.02,
        hierarchy_nmax=1,
        tamen_tolerance=2e-3,
        trace_tolerance=2e-3,
        hermiticity_tolerance=2e-3,
        electron_number_tolerance=2e-3,
        tt_max_rank=1,
    )
    bath_model = build_quartic_model(bath_config, closed_mode, quartic_test_fit())
    bath_result = @test_logs (:warn, r"TT rank") propagate_quartic_heom(
        bath_model,
        bath_config,
    )
    @test length(bath_result.times) == 3
    @test all(isfinite, bath_result.populations)
    @test all(value -> isfinite(real(value)) && isfinite(imag(value)), bath_result.trace)
    @test maximum(abs.(bath_result.trace .- 1)) < bath_config.trace_tolerance
    @test maximum(bath_result.hermiticity_error) < bath_config.hermiticity_tolerance
    @test maximum(abs.(bath_result.electron_number .- 1)) <
          bath_config.electron_number_tolerance
end

@testset "quartic executable sources are import safe" begin
    cairo_was_loaded = isdefined(@__MODULE__, :CairoMakie)
    mktempdir() do directory
        stdout_path = joinpath(directory, "stdout.txt")
        stderr_path = joinpath(directory, "stderr.txt")
        open(stdout_path, "w") do output
            open(stderr_path, "w") do error_output
                cd(directory) do
                    redirect_stdout(output) do
                        redirect_stderr(error_output) do
                            include(joinpath(@__DIR__, "..", "one_site_relaxation.jl"))
                            include(joinpath(@__DIR__, "..", "two_site_transfer.jl"))
                        end
                    end
                end
            end
        end
        @test read(stdout_path, String) == ""
        @test read(stderr_path, String) == ""
        @test sort(readdir(directory)) == ["stderr.txt", "stdout.txt"]
    end
    @test isdefined(@__MODULE__, :CairoMakie) == cairo_was_loaded
end

@testset "one-site quartic workflow uses injected boundaries" begin
    config = QuarticConfig(
        site_count=1,
        site_energies=[0.0],
        hopping=0.0,
        d_raw=4,
        d_keep=2,
        bath_lambda=0.1,
        hierarchy_nmax=1,
        final_time=0.1,
        time_step=0.1,
    )
    fit = quartic_test_fit()
    correlation_calls = Ref(0)
    propagation_calls = Ref(0)
    plot_calls = Ref(0)
    correlation_builder = _ -> begin
        correlation_calls[] += 1
        fit
    end
    propagator = (model, propagated_config) -> begin
        propagation_calls[] += 1
        @test propagated_config === config
        quartic_test_zero_result(model, propagated_config)
    end
    plotter = (paths, result; overwrite=false) -> begin
        plot_calls[] += 1
        @test basename(paths.oscillator_q) == "one_site_q.png"
        @test !overwrite
        nothing
    end

    mktempdir() do directory
        progress = IOBuffer()
        workflow = one_site_main(
            config;
            output_directory=directory,
            correlation_builder,
            propagator,
            plotter,
            progress_io=progress,
        )
        @test workflow.result.times == [0.0]
        @test workflow.fit === fit
        @test workflow.config === config
        @test isfile(workflow.paths.csv)
        @test correlation_calls[] == 1
        @test propagation_calls[] == 1
        @test plot_calls[] == 1
        diagnostics = String(take!(progress))
        for label in (
            "rates",
            "coeff_forward",
            "coeff_backward",
            "training_absolute_error",
            "training_relative_error",
            "validation_absolute_error",
            "validation_relative_error",
            "vandermonde_condition",
            "minimum_pole_separation",
        )
            @test occursin(label, diagnostics)
        end
    end
end

@testset "two-site workflow reuses one finite-bath fit across four cases" begin
    config = QuarticConfig(
        site_count=2,
        site_energies=[0.1, -0.1],
        hopping=0.2,
        d_raw=4,
        d_keep=2,
        K4=0.15,
        bath_lambda=0.1,
        hierarchy_nmax=1,
        final_time=0.1,
        time_step=0.1,
    )
    fit = quartic_test_fit()
    correlation_calls = Ref(0)
    propagation_calls = Ref(0)
    comparison_plot_calls = Ref(0)
    correlation_builder = _ -> begin
        correlation_calls[] += 1
        fit
    end
    propagator = (model, propagated_config) -> begin
        propagation_calls[] += 1
        quartic_test_zero_result(model, propagated_config)
    end
    plotter = (path, cases; overwrite=false) -> begin
        comparison_plot_calls[] += 1
        @test basename(path) == "two_site_population_comparison.png"
        @test length(cases) == 4
        @test !overwrite
        nothing
    end

    mktempdir() do directory
        workflow = two_site_main(
            config;
            output_directory=directory,
            correlation_builder,
            propagator,
            plotter,
            progress_io=IOBuffer(),
        )
        labels = getfield.(workflow.cases, :label)
        @test labels == [
            "harmonic_no_bath",
            "harmonic_bath",
            "quartic_no_bath",
            "quartic_bath",
        ]
        @test getfield.(workflow.cases, :fit) == [nothing, fit, nothing, fit]
        @test getfield.(getfield.(workflow.cases, :config), :bath_lambda) ==
              [0.0, config.bath_lambda, 0.0, config.bath_lambda]
        @test getfield.(getfield.(workflow.cases, :config), :K4) ==
              [0.0, 0.0, config.K4, config.K4]
        @test all(case -> isfile(case.paths.csv), workflow.cases)
        @test isfile(workflow.metadata_path)
        metadata = read(workflow.metadata_path, String)
        @test occursin("harmonic_no_bath", metadata)
        @test occursin("quartic_bath", metadata)
        @test occursin("coeff_forward", metadata)
        @test correlation_calls[] == 1
        @test propagation_calls[] == 4
        @test comparison_plot_calls[] == 1
    end
end

@testset "quartic workflows preflight outputs before expensive work" begin
    one_site_config = QuarticConfig(
        site_count=1,
        site_energies=[0.0],
        hopping=0.0,
        d_raw=4,
        d_keep=2,
        final_time=0.1,
        time_step=0.1,
    )
    two_site_config = quartic_test_config_with(
        one_site_config;
        site_count=2,
        site_energies=[0.0, 0.0],
    )
    builder_calls = Ref(0)
    forbidden_builder = _ -> begin
        builder_calls[] += 1
        error("correlation builder must not run after failed output preflight")
    end

    mktempdir() do directory
        write(joinpath(directory, "one_site.csv"), "existing")
        @test_throws ArgumentError one_site_main(
            one_site_config;
            output_directory=directory,
            correlation_builder=forbidden_builder,
            progress_io=IOBuffer(),
        )
    end
    mktempdir() do directory
        write(joinpath(directory, "two_site_metadata.txt"), "existing")
        @test_throws ArgumentError two_site_main(
            two_site_config;
            output_directory=directory,
            correlation_builder=forbidden_builder,
            progress_io=IOBuffer(),
        )
    end
    @test builder_calls[] == 0
end

@testset "quartic plot publication is atomic without loading CairoMakie" begin
    cairo_was_loaded = isdefined(@__MODULE__, :CairoMakie)
    mktempdir() do directory
        path = joinpath(directory, "diagnostic.png")
        @test _quartic_publish_png(path) do temporary_path
            write(temporary_path, "png fixture")
        end == path
        @test read(path, String) == "png fixture"
        @test_throws ArgumentError _quartic_publish_png(path) do temporary_path
            write(temporary_path, "replacement")
        end
        @test read(path, String) == "png fixture"
    end
    @test isdefined(@__MODULE__, :CairoMakie) == cairo_was_loaded
end

@testset "quartic comparison plot lazily renders with CairoMakie" begin
    result = (
        times=[0.0, 1.0],
        populations=[1.0 0.5; 0.0 0.5],
    )
    cases = [(label="smoke", result=result)]
    mktempdir() do directory
        path = joinpath(directory, "comparison.png")
        @test plot_two_site_comparison(path, cases) == path
        @test isfile(path)
        @test filesize(path) > 0
    end
end
