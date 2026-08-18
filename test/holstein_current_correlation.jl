using LinearAlgebra
using Statistics
using Test
using TTDynamics
using TTSolver
using KaisouEOM

if !isdefined(@__MODULE__, :HolsteinConfig)
    include("../examples/holstein/utils.jl")
end
include("../examples/holstein_current_correlation/utils.jl")
include("../examples/holstein_current_correlation/equilibrate.jl")

function current_correlation_problem(config)
    H = periodic_holstein_hamiltonian(config.site_energies_cm, config.hopping_cm) *
        KaisouEOM.icm2ifs
    baths = [
        BathExp(ComplexF64[0.4], ComplexF64[0.03 - 0.02im], projector)
        for projector in site_projectors(config.site_count)
    ]
    system = HEOMTTSystem(H, NoiseExp(baths), config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable, population_observables)
end

function unchecked_holstein_config(config; site_count=config.site_count,
                                   hopping_cm=config.hopping_cm)
    values = Any[getfield(config, index) for index in 1:fieldcount(HolsteinConfig)]
    values[1] = site_count
    values[3] = hopping_cm
    return HolsteinConfig(values...)
end

function metadata_error(f)
    try
        f()
    catch error
        @test error isa ArgumentError
        return error
    end
    error("expected ArgumentError")
end

function metadata_fixture()
    config = HolsteinConfig(
        site_count=2,
        site_energies_cm=[10.0, -5.0],
        hopping_cm=2.0,
        final_time_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
    )
    problem = current_correlation_problem(config)
    decomposition = (
        exponents=ComplexF64[0.1 + 0.2im, 0.3 - 0.4im],
        coefficients=ComplexF64[0.5 - 0.6im, 0.7 + 0.8im],
    )
    state = build_initial_state(problem.system, config.initial_site; tol=1e-14)
    metadata = equilibrium_metadata(
        config,
        decomposition,
        problem,
        state;
        equilibration_time_fs=1000.0,
    )
    return (; config, decomposition, problem, state, metadata)
end

function equilibration_fixture()
    config = HolsteinConfig(
        site_count=2,
        site_energies_cm=zeros(2),
        hopping_cm=2.0,
        final_time_fs=1.0,
        time_step_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
        state_rounding_tolerance=1e-14,
        tamen_tolerance=1e-12,
    )
    full_problem = current_correlation_problem(config)
    state = build_initial_state(full_problem.system, config.initial_site; tol=1e-14)
    zero_liouvillian = TTMatrix([
        zeros(ComplexF64, 1, dimension, dimension, 1)
        for dimension in tt_dims(state)
    ])
    problem = (
        system=full_problem.system,
        liouvillian=zero_liouvillian,
        trace_observable=full_problem.trace_observable,
        population_observables=full_problem.population_observables,
    )
    decomposition = (
        exponents=ComplexF64[0.1 + 0.2im],
        coefficients=ComplexF64[0.5 - 0.6im],
    )
    return (; config, decomposition, problem, state)
end

@testset "Holstein current correlation utilities" begin
    config = HolsteinConfig(
        site_count=3,
        site_energies_cm=zeros(3),
        hopping_cm=2.0,
        final_time_fs=1.0,
    )
    current = periodic_current_operator(config)
    scale = config.hopping_cm * KaisouEOM.icm2ifs

    @test ishermitian(current)
    @test current[2, 1] == 1im * scale
    @test current[1, 2] == -1im * scale
    @test current[1, 3] == 1im * scale
    @test current[3, 1] == -1im * scale
    @test diag(current) == zeros(3)

    config2 = HolsteinConfig(
        site_count=2,
        site_energies_cm=zeros(2),
        hopping_cm=2.0,
        final_time_fs=1.0,
    )
    current2 = periodic_current_operator(config2)
    @test current2 == ComplexF64[0 -1im * scale; 1im * scale 0]
    @test_throws ArgumentError periodic_current_operator(
        unchecked_holstein_config(config; site_count=1),
    )
    @test_throws ArgumentError periodic_current_operator(
        unchecked_holstein_config(config; hopping_cm=-1.0),
    )
    @test_throws ArgumentError periodic_current_operator(
        unchecked_holstein_config(config; hopping_cm=NaN),
    )
end

@testset "Holstein equilibrium metadata" begin
    fixture = metadata_fixture()
    (; config, decomposition, problem, state, metadata) = fixture

    @test metadata["identifier"] == "TTDynamics.HolsteinEquilibrium"
    @test metadata["version"] == 1
    @test metadata["heom_representation"] == "twin-space-v1"
    @test metadata["equilibration_time_fs"] == 1000.0
    for field in fieldnames(HolsteinConfig)
        expected = field === :pade_type ? string(config.pade_type) : getfield(config, field)
        @test metadata[String(field)] == expected
    end
    @test metadata["site_count"] == config.site_count
    @test metadata["site_energies_cm"] == config.site_energies_cm
    @test metadata["hopping_cm"] == config.hopping_cm
    @test metadata["brownian_frequency_cm"] == config.brownian_frequency_cm
    @test metadata["brownian_damping_cm"] == config.brownian_damping_cm
    @test metadata["reorganization_energy_cm"] == config.reorganization_energy_cm
    @test metadata["temperature_K"] == config.temperature_K
    @test metadata["time_step_fs"] == config.time_step_fs
    @test metadata["pade_order"] == config.pade_order
    @test metadata["pade_type"] == string(config.pade_type)
    @test metadata["tpsd_tolerance"] == config.tpsd_tolerance
    @test metadata["hierarchy_local_size"] == config.hierarchy_local_size
    @test metadata["exponents_real"] == real.(decomposition.exponents)
    @test metadata["exponents_imag"] == imag.(decomposition.exponents)
    @test metadata["coefficients_real"] == real.(decomposition.coefficients)
    @test metadata["coefficients_imag"] == imag.(decomposition.coefficients)
    @test metadata["hierarchy_sizes"] == problem.system.nb
    @test metadata["tt_dimensions"] == tt_dims(state)

    mktempdir() do directory
        path = joinpath(directory, "equilibrium.toml")
        @test write_equilibrium_metadata(path, metadata) == path
        @test read_equilibrium_metadata(path) == metadata
        @test_throws ArgumentError write_equilibrium_metadata(path, metadata)
        replacement = deepcopy(metadata)
        replacement["equilibration_time_fs"] = 500.0
        @test write_equilibrium_metadata(path, replacement; overwrite=true) == path
        @test read_equilibrium_metadata(path) == replacement
        missing_error = metadata_error(() -> read_equilibrium_metadata(
            joinpath(directory, "missing.toml"),
        ))
        @test occursin("metadata file", sprint(showerror, missing_error))
    end

    @test validate_equilibrium_state(state, metadata, config, decomposition, problem) === state

    for (field, replacement) in (
        "heom_representation" => "vectorized-v0",
        "hopping_cm" => 3.0,
        "exponents_imag" => [0.2, -0.3],
        "hierarchy_sizes" => [3, 2],
        "tt_dimensions" => [2, 2, 3, 2],
        "identifier" => "unsupported",
        "version" => 2,
        "coefficients_real" => [NaN, 0.7],
    )
        changed = deepcopy(metadata)
        changed[field] = replacement
        error = metadata_error(() -> validate_equilibrium_state(
            state,
            changed,
            config,
            decomposition,
            problem,
        ))
        @test occursin(field, sprint(showerror, error))
    end

    old_layout = TTTensor([
        reshape(ComplexF64[1, 0, 0, 0], 1, 4, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
    ])
    old_error = metadata_error(() -> validate_equilibrium_state(
        old_layout,
        metadata,
        config,
        decomposition,
        problem,
    ))
    @test occursin("old vectorized HEOM state is unsupported", sprint(showerror, old_error))

    mktempdir() do directory
        state_path = joinpath(directory, "equilibrium.ttbin")
        metadata_path = joinpath(directory, "equilibrium.toml")
        save_tt_binary(state_path, state)
        write_equilibrium_metadata(metadata_path, metadata)
        restored = load_tt_binary(state_path)
        @test validate_equilibrium_state(
            restored,
            read_equilibrium_metadata(metadata_path),
            config,
            decomposition,
            problem,
        ).cores == state.cores

        matrix = TTMatrix([
            reshape(ComplexF64[1], 1, 1, 1, 1),
        ])
        save_tt_binary(state_path, matrix; overwrite=true)
        error = metadata_error(() -> validate_equilibrium_state(
            load_tt_binary(state_path),
            metadata,
            config,
            decomposition,
            problem,
        ))
        @test occursin("TTTensor", sprint(showerror, error))
    end
end

@testset "Holstein current twin-space operators" begin
    config = HolsteinConfig(
        site_count=2,
        site_energies_cm=zeros(2),
        hopping_cm=2.0,
        final_time_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
    )
    problem = current_correlation_problem(config)
    operators = build_current_heom_operators(config, problem)
    current = operators.current
    rho = ComplexF64[0.2 + 0.3im -0.1 + 0.4im; 0.7 - 0.2im 0.8 - 0.5im]
    h1 = ComplexF64[1 + 0.2im, -0.3 + 0.4im]
    h2 = ComplexF64[-0.4 + 0.7im, 0.9 - 0.1im]
    ket_core = reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2)
    bra_core = reshape(rho, 2, 2, 1)
    state = TTTensor([
        ket_core,
        bra_core,
        reshape(h1, 1, 2, 1),
        reshape(h2, 1, 2, 1),
    ])
    source = operators.left_action * state
    # `tt_full` stores the first TT mode as the final dense axis.  Its column-
    # major flattening therefore places bra and hierarchy indices before ket.
    expected = kron(vec(transpose(current * rho)), kron(h1, h2))
    @test vec(tt_full(source)) ≈ expected

    root_state = TTTensor([
        ket_core,
        bra_core,
        reshape(ComplexF64[1, 0], 1, 2, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
    ])
    @test tt_dot(operators.observable, root_state) ≈ tr(current * rho)

    auxiliary_only_state = TTTensor([
        ket_core,
        bra_core,
        reshape(ComplexF64[0, 1], 1, 2, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
    ])
    @test tt_dot(operators.observable, auxiliary_only_state) == 0
end

@testset "Holstein state propagation and measurements" begin
    config = HolsteinConfig(
        site_count=2,
        site_energies_cm=zeros(2),
        hopping_cm=2.0,
        final_time_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
        state_rounding_tolerance=1e-14,
        tamen_tolerance=1e-12,
    )
    rho = TTTensor([
        reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2),
        reshape(ComplexF64[1 2im; -3im 4], 2, 2, 1),
    ])
    zero_liouvillian = TTMatrix([
        zeros(ComplexF64, 1, 2, 2, 1),
        reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2, 1),
    ])
    @test tt_full(propagate_cn_step(rho, zero_liouvillian, config)) ≈ tt_full(rho)
    trace_observable = TTTensor([
        reshape(ComplexF64[1, 0], 1, 2, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
    ])
    @test tt_full(propagate_cn_step(
        rho,
        zero_liouvillian,
        config;
        trace_observable,
    )) ≈ tt_full(rho)

    traced_observations = propagate_fixed_steps(
        rho,
        (; liouvillian=zero_liouvillian, trace_observable),
        config,
        1;
        observe=(step, time, state) -> (; step, time, state),
    )
    @test tt_full(traced_observations.state) ≈ tt_full(rho)
    @test [observation.step for observation in traced_observations.observations] == [0, 1]

    observations = propagate_fixed_steps(
        rho,
        (; liouvillian=zero_liouvillian),
        config,
        1;
        observe=(step, time, state) -> (; step, time, norm=norm(tt_full(state))),
    )
    @test tt_full(observations.state) ≈ tt_full(rho)
    @test [observation.step for observation in observations.observations] == [0, 1]
    @test [observation.time for observation in observations.observations] == [0.0, config.time_step_fs]
    @test_throws ArgumentError propagate_fixed_steps(rho, (; liouvillian=zero_liouvillian), config, -1; observe=(args...) -> nothing)

    problem = current_correlation_problem(config)
    scaled = (2 - 3im) * build_initial_state(problem.system, 1; tol=1e-14)
    normalized = normalize_heom_state(scaled, problem.trace_observable)
    @test tt_dot(problem.trace_observable, normalized) ≈ 1
    @test tt_full(normalized) ≈ tt_full(scaled) / (2 - 3im)
    zero_state = TTTensor([
        zeros(ComplexF64, 1, 2, 1),
        zeros(ComplexF64, 1, 2, 1),
        zeros(ComplexF64, 1, 2, 1),
        zeros(ComplexF64, 1, 2, 1),
    ])
    @test_throws ArgumentError normalize_heom_state(zero_state, problem.trace_observable)

    measurement = measure_heom_state(
        build_initial_state(problem.system, 1; tol=1e-14),
        problem.trace_observable,
        problem.population_observables,
    )
    @test measurement.populations ≈ [1.0, 0.0]
    @test measurement.trace ≈ 1.0
    @test measurement.maximum_rank >= 1
    @test measurement.mean_rank >= 1
end

@testset "Holstein fixed-time equilibration" begin
    fixture = equilibration_fixture()
    (; config, decomposition, problem, state) = fixture

    @test DEFAULT_CONFIG isa HolsteinConfig
    result = run_equilibration(
        config,
        problem;
        initial_state=state,
        equilibration_time_fs=config.time_step_fs,
    )
    @test result.times == [0.0, 1.0]
    @test size(result.populations) == (config.site_count, 2)
    @test length(result.trace) == length(result.times)
    @test length(result.maximum_rank) == length(result.times)
    @test length(result.mean_rank) == length(result.times)
    @test result.state isa TTTensor
    @test tt_dot(problem.trace_observable, result.state) ≈ 1
    @test_throws ArgumentError run_equilibration(
        config,
        problem;
        initial_state=state,
        equilibration_time_fs=1.5,
    )
    @test_throws ArgumentError run_equilibration(
        config,
        problem;
        initial_state=state,
        equilibration_time_fs=0.0,
    )

    mktempdir() do directory
        outputs = save_equilibration_outputs(
            config,
            decomposition,
            problem,
            result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
        )
        @test isfile(outputs.state_path)
        @test isfile(outputs.metadata_path)
        @test isfile(outputs.csv_path)
        @test_throws ArgumentError write_equilibration_csv(outputs.csv_path, result)
        restored = load_tt_binary(outputs.state_path)
        @test validate_equilibrium_state(
            restored,
            read_equilibrium_metadata(outputs.metadata_path),
            config,
            decomposition,
            problem,
        ).cores == result.state.cores
        @test_throws ArgumentError save_equilibration_outputs(
            config,
            decomposition,
            problem,
            result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
        )
    end

    mktempdir() do directory
        main_result = equilibrate_main(
            config;
            equilibration_time_fs=config.time_step_fs,
            output_directory=directory,
            decomposition_builder=_ -> decomposition,
            problem_builder=(_, _) -> problem,
            equilibration_runner=(args...; kwargs...) -> result,
        )
        @test isfile(main_result.state_path)
        @test isfile(main_result.metadata_path)
        @test isfile(main_result.csv_path)
    end
end

@testset "Holstein reloaded current correlation" begin
    fixture = equilibration_fixture()
    (; config, problem, state) = fixture
    operators = build_current_heom_operators(config, problem)

    result = run_current_correlation(
        state,
        config,
        problem;
        correlation_time_fs=0.0,
    )
    source = operators.left_action * state
    expected = tt_dot(operators.observable, source)
    @test result.times == [0.0]
    @test result.correlation == ComplexF64[expected]
    @test result.maximum_rank == [maximum(tt_ranks(result.state))]
    @test result.mean_rank == [mean(tt_ranks(result.state))]
    @test_throws ArgumentError run_current_correlation(
        state,
        config,
        problem;
        correlation_time_fs=0.5,
    )

    synthetic_result = (
        times=[0.0, 1.0],
        correlation=ComplexF64[1.5 - 2.0im, -3.0 + 4.25im],
        maximum_rank=[2, 3],
        mean_rank=[1.5, 2.0],
    )
    mktempdir() do directory
        path = joinpath(directory, "current_correlation.csv")
        @test write_current_correlation_csv(path, synthetic_result) == path
        rows = readlines(path)
        @test rows[1] == "time_fs,correlation_real_fs^-2,correlation_imag_fs^-2,max_rank,mean_rank"
        @test rows[2] == "0.0,1.5,-2.0,2,1.5"
        @test rows[3] == "1.0,-3.0,4.25,3,2.0"
        @test_throws ArgumentError write_current_correlation_csv(path, synthetic_result)
    end
end
