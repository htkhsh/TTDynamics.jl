using LinearAlgebra
using Test
using TTDynamics
using TTSolver
using HEOMKit

include("../examples/lattice_frohlich_current_correlation/config.jl")
include("../examples/lattice_frohlich_current_correlation/model.jl")
include("../examples/lattice_frohlich_current_correlation/utils.jl")
include("../examples/lattice_frohlich_current_correlation/equilibrate.jl")
include("../examples/lattice_frohlich_current_correlation/current_correlation.jl")

@testset "Lattice Frohlich current configuration" begin
    config = LatticeFrohlichCurrentCorrelationConfig()
    expected = (
        site_count=5, site_energies_cm=zeros(5), hopping_cm=400.0,
        brownian_frequency_cm=1400.0, brownian_damping_cm=200.0,
        reorganization_energy_cm=600.0, temperature_K=300.0,
        initial_site=1, final_time_fs=500.0, time_step_fs=1.0,
        pade_order=8, tpsd_tolerance=2e-2, pade_type=:Nm1,
        validation_final_time_fs=100.0, validation_sample_count=200,
        bcf_upper_bound_cm=10_000.0, hierarchy_local_size=4,
        temporal_basis_size=3, tamen_tolerance=2e-2,
        operator_tolerance=1e-10, state_rounding_tolerance=1e-10,
        sweep_count=3, local_iterations=5, kick_rank=4,
        progress_interval=10,
    )
    @test fieldnames(typeof(config)) == keys(expected)
    @test all(field -> getproperty(config, field) == getproperty(expected, field), keys(expected))
    invalid_configurations = (
        "site count" => () -> LatticeFrohlichCurrentCorrelationConfig(site_count=1),
        "site energy count" => () -> LatticeFrohlichCurrentCorrelationConfig(
            site_count=3, site_energies_cm=zeros(2),
        ),
        "finite site energies" => () -> LatticeFrohlichCurrentCorrelationConfig(
            site_energies_cm=[0.0, NaN, 0.0, 0.0, 0.0],
        ),
        "initial site" => () -> LatticeFrohlichCurrentCorrelationConfig(initial_site=0),
        "nonnegative hopping" => () -> LatticeFrohlichCurrentCorrelationConfig(
            hopping_cm=-1.0,
        ),
        "finite scales" => () -> LatticeFrohlichCurrentCorrelationConfig(
            temperature_K=Inf,
        ),
        "positive physical scales" => () -> LatticeFrohlichCurrentCorrelationConfig(
            brownian_frequency_cm=0.0,
        ),
        "positive numerical scales" => () -> LatticeFrohlichCurrentCorrelationConfig(
            operator_tolerance=0.0,
        ),
        "underdamped Brownian poles" => () -> LatticeFrohlichCurrentCorrelationConfig(
            brownian_frequency_cm=100.0, brownian_damping_cm=200.0,
        ),
        "positive Pade order" => () -> LatticeFrohlichCurrentCorrelationConfig(pade_order=0),
        "Pade type" => () -> LatticeFrohlichCurrentCorrelationConfig(pade_type=:invalid),
        "validation samples" => () -> LatticeFrohlichCurrentCorrelationConfig(
            validation_sample_count=1,
        ),
        "solver controls" => () -> LatticeFrohlichCurrentCorrelationConfig(
            hierarchy_local_size=0,
        ),
        "temporal basis" => () -> LatticeFrohlichCurrentCorrelationConfig(
            temporal_basis_size=4,
        ),
        "integral time grid" => () -> LatticeFrohlichCurrentCorrelationConfig(
            final_time_fs=1.5, time_step_fs=1.0,
        ),
    )
    for (description, construct) in invalid_configurations
        @testset "$description" begin
            @test_throws ArgumentError construct()
        end
    end
end

@testset "Lattice Frohlich equilibrium checkpoint publication" begin
    config = LatticeFrohlichCurrentCorrelationConfig(
        site_count=3, site_energies_cm=zeros(3), hopping_cm=20.0,
        final_time_fs=5.0, time_step_fs=1.0, hierarchy_local_size=2,
        operator_tolerance=1e-14, state_rounding_tolerance=1e-14,
        tamen_tolerance=1e-12,
    )
    decomposition = (
        exponents=ComplexF64[0.4],
        coefficients=ComplexF64[0.03 - 0.02im],
    )
    full_problem = build_lattice_frohlich_current_model(config, decomposition)
    state = build_initial_state(full_problem.system, config.initial_site; tol=1e-14)
    metadata = lattice_frohlich_equilibrium_metadata(
        config,
        decomposition,
        full_problem,
        state;
        equilibration_time_fs=config.time_step_fs,
    )
    @test metadata["identifier"] == "TTDynamics.LatticeFrohlichEquilibrium"
    @test metadata["version"] == 1
    @test metadata["heom_representation"] == "twin-space-v1"
    @test metadata["frohlich_weights"] == [collect(row) for row in eachrow(full_problem.frohlich_weights)]
    for field in fieldnames(LatticeFrohlichCurrentCorrelationConfig)
        expected = field === :pade_type ? string(getfield(config, field)) : getfield(config, field)
        @test metadata[String(field)] == expected
    end

    mktempdir() do directory
        metadata_path = joinpath(directory, "equilibrium.toml")
        state_path = joinpath(directory, "equilibrium.ttbin")
        @test write_lattice_frohlich_equilibrium_metadata(metadata_path, metadata) == metadata_path
        @test read_lattice_frohlich_equilibrium_metadata(metadata_path) == metadata
        save_tt_binary(state_path, state)
        restored = load_tt_binary(state_path)
        @test validate_lattice_frohlich_equilibrium_state(
            restored, metadata, config, decomposition, full_problem,
        ).cores == state.cores
    end

    for (field, replacement) in (
        "frohlich_weights" => [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "exponents_real" => [0.5],
        "hierarchy_sizes" => [3],
        "identifier" => "other",
        "hopping_cm" => config.hopping_cm + 1.0,
        "coefficients_imag" => [imag(first(decomposition.coefficients)) + 0.1],
        "version" => 2,
        "heom_representation" => "vectorized-v0",
        "tt_dimensions" => [3, 3, 2, 2, 3],
    )
        changed = deepcopy(metadata)
        changed[field] = replacement
        @test_throws ArgumentError validate_lattice_frohlich_equilibrium_state(
            state, changed, config, decomposition, full_problem,
        )
    end

    wrong_dimension_state = TTTensor([
        reshape(Matrix{ComplexF64}(I, 3, 3), 1, 3, 3),
        reshape(ComplexF64[1 0 0; 0 0 0; 0 0 0], 3, 3, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
        reshape(ComplexF64[1, 0, 0], 1, 3, 1),
    ])
    @test_throws ArgumentError validate_lattice_frohlich_equilibrium_state(
        wrong_dimension_state, metadata, config, decomposition, full_problem,
    )
end

@testset "Lattice Frohlich equilibration output orchestration" begin
    config = LatticeFrohlichCurrentCorrelationConfig(
        site_count=3, site_energies_cm=zeros(3), hopping_cm=20.0,
        final_time_fs=5.0, time_step_fs=1.0, hierarchy_local_size=2,
        operator_tolerance=1e-14, state_rounding_tolerance=1e-14,
        tamen_tolerance=1e-12,
    )
    decomposition = (
        exponents=ComplexF64[0.4],
        coefficients=ComplexF64[0.03 - 0.02im],
    )
    full_problem = build_lattice_frohlich_current_model(config, decomposition)
    state = build_initial_state(full_problem.system, config.initial_site; tol=1e-14)
    zero_liouvillian = TTMatrix([
        zeros(ComplexF64, 1, dimension, dimension, 1)
        for dimension in tt_dims(state)
    ])
    problem = (; system=full_problem.system, liouvillian=zero_liouvillian,
               trace_observable=full_problem.trace_observable,
               population_observables=full_problem.population_observables)
    result = run_lattice_frohlich_equilibration(
        config, problem;
        initial_state=state,
        equilibration_time_fs=config.time_step_fs,
    )
    @test _lattice_frohlich_equilibration_output_paths("outputs") == (
        state_path="outputs/lattice_frohlich_equilibrium.ttbin",
        metadata_path="outputs/lattice_frohlich_equilibrium_metadata.toml",
        csv_path="outputs/lattice_frohlich_equilibration.csv",
        png_path="outputs/lattice_frohlich_equilibration_populations.png",
    )

    mktempdir() do directory
        plotter = (path, _) -> write(path, "synthetic PNG")
        saved = save_lattice_frohlich_equilibration_outputs(
            config, decomposition, full_problem, result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
            plotter,
        )
        @test all(isfile, values(saved))
        @test tt_dot(full_problem.trace_observable, load_tt_binary(saved.state_path)) ≈ 1
        @test occursin("time_fs,population_site_1", read(saved.csv_path, String))
        @test_throws ArgumentError save_lattice_frohlich_equilibration_outputs(
            config, decomposition, full_problem, result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
            plotter,
        )
        replaced = save_lattice_frohlich_equilibration_outputs(
            config, decomposition, full_problem, result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
            overwrite=true,
            plotter,
        )
        @test replaced == saved
    end

    mktempdir() do directory
        failing_metadata_writer = (args...; kwargs...) -> error("forced metadata failure")
        paths = _lattice_frohlich_equilibration_output_paths(directory)
        @test_throws ErrorException save_lattice_frohlich_equilibration_outputs(
            config, decomposition, full_problem, result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
            metadata_writer=failing_metadata_writer,
            plotter=(path, _) -> write(path, "synthetic PNG"),
        )
        @test all(path -> !ispath(path), values(paths))
    end

    mktempdir() do directory
        progress_io = IOBuffer()
        invoked = Ref(false)
        runner = function(main_config, main_problem; progress_callback, kwargs...)
            invoked[] = true
            progress_callback((;
                step=0, step_count=1, time_fs=0.0, trace=1.0,
                maximum_rank=1, mean_rank=1.0, elapsed_seconds=0.0,
            ))
            return result
        end
        main_result = lattice_frohlich_equilibrate_main(
            config;
            equilibration_time_fs=config.time_step_fs,
            output_directory=directory,
            decomposition_builder=_ -> decomposition,
            problem_builder=(_, _) -> full_problem,
            equilibration_runner=runner,
            plotter=(path, _) -> write(path, "synthetic PNG"),
            progress_io,
        )
        @test invoked[]
        @test all(isfile, values(main_result.paths))
        @test occursin("Step 0/1", String(take!(progress_io)))
    end
end

@testset "Lattice Frohlich current model" begin
    @test periodic_lattice_frohlich_distance(1, 5, 5) == 1
    @test periodic_lattice_frohlich_distance(1, 4, 5) == 2
    @test periodic_lattice_frohlich_distance(1, 3, 4) == 2
    @test periodic_lattice_frohlich_distance(1, 4, 4) == 1
    @test periodic_lattice_frohlich_distance(1, 2, 2) == 1

    @test periodic_lattice_frohlich_current_hamiltonian([1.0, 2.0], 3.0) ==
          ComplexF64[1 -3; -3 2]
    @test periodic_lattice_frohlich_current_hamiltonian([1.0, 2.0, 3.0, 4.0], 5.0) ==
          ComplexF64[1 -5 0 -5; -5 2 -5 0; 0 -5 3 -5; -5 0 -5 4]

    weights = normalized_lattice_frohlich_current_kernel(4)
    @test size(weights) == (4, 4)
    @test all(n -> sum(abs2, weights[:, n]) ≈ 1, 1:4)
    @test weights ≈ transpose(weights)
    @test weights[:, 2] ≈ circshift(weights[:, 1], 1)
    @test_throws ArgumentError normalized_lattice_frohlich_current_kernel(
        3; kernel=_ -> -1.0,
    )
    @test_throws ArgumentError normalized_lattice_frohlich_current_kernel(
        3; kernel=_ -> Inf,
    )
    @test_throws ArgumentError normalized_lattice_frohlich_current_kernel(
        3; kernel=_ -> 0.0,
    )
    operators = lattice_frohlich_current_coupling_operators(4)
    @test all(isdiag, operators)
    @test all(ishermitian, operators)
    @test all(1:4) do bath
        diag(operators[mod1(bath + 1, 4)]) ≈ circshift(diag(operators[bath]), 1)
    end

    bath_config = LatticeFrohlichCurrentCorrelationConfig(
        site_count=2,
        site_energies_cm=zeros(2),
        final_time_fs=1.0,
        pade_order=1,
        tpsd_tolerance=0.1,
        validation_final_time_fs=1.0,
        validation_sample_count=3,
        hierarchy_local_size=2,
    )
    decomposition = decompose_lattice_frohlich_current_bath(bath_config)
    @test !isempty(decomposition.exponents)
    @test length(decomposition.exponents) == length(decomposition.coefficients)
    @test all(isfinite, decomposition.exponents)
    @test all(isfinite, decomposition.coefficients)
    @test all(>(0), real.(decomposition.exponents))
    @test isfinite(decomposition.relative_error)
    @test decomposition.relative_error <= bath_config.tpsd_tolerance
end

function lattice_frohlich_current_problem(config)
    decomposition = (
        exponents=ComplexF64[0.4],
        coefficients=ComplexF64[0.03 - 0.02im],
    )
    return build_lattice_frohlich_current_model(config, decomposition)
end

function lattice_frohlich_current_fixture()
    config = LatticeFrohlichCurrentCorrelationConfig(
        site_count=3,
        site_energies_cm=zeros(3),
        hopping_cm=20.0,
        final_time_fs=5.0,
        time_step_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
        state_rounding_tolerance=1e-14,
        tamen_tolerance=1e-12,
    )
    full_problem = lattice_frohlich_current_problem(config)
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
    return (; config, problem, state)
end

@testset "Lattice Frohlich particle current" begin
    config = LatticeFrohlichCurrentCorrelationConfig(
        site_count=3, site_energies_cm=zeros(3), hopping_cm=20.0,
        final_time_fs=2.0, hierarchy_local_size=2,
    )
    current = lattice_frohlich_particle_current(config)
    scale = 20.0 * HEOMKit.icm2ifs
    @test current == ComplexF64[0 -im*scale im*scale;
                               im*scale 0 -im*scale;
                               -im*scale im*scale 0]
    @test ishermitian(current)
    hermiticity_error = try
        _validate_lattice_frohlich_particle_current(ComplexF64[0 1; 0 0])
        nothing
    catch error
        error
    end
    @test hermiticity_error isa ErrorException
    @test occursin("Hermitian", sprint(showerror, hermiticity_error))

    problem = lattice_frohlich_current_problem(config)
    operators = build_lattice_frohlich_current_operators(config, problem)
    @test tt_dims(operators.left_action) == tt_dims(problem.liouvillian)
    @test tt_dims(operators.observable) == heom_tt_dimensions(problem.system)

    exact_config = LatticeFrohlichCurrentCorrelationConfig(
        site_count=2,
        site_energies_cm=zeros(2),
        hopping_cm=20.0,
        final_time_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
        state_rounding_tolerance=1e-14,
        tamen_tolerance=1e-12,
    )
    exact_problem = lattice_frohlich_current_problem(exact_config)
    exact_operators = build_lattice_frohlich_current_operators(exact_config, exact_problem)
    rho = ComplexF64[0.7 0.2 + 0.1im; -0.3 + 0.4im 0.3]
    root_state = TTTensor([
        reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2),
        reshape(rho, 2, 2, 1),
        [reshape(ComplexF64[1; zeros(n - 1)], 1, n, 1)
         for n in exact_problem.system.nb]...,
    ])
    source = exact_operators.left_action * root_state
    @test root_density_matrix(source, exact_problem.system) ≈ exact_operators.current * rho
    @test tt_dot(exact_operators.observable, source) ≈
          tr(exact_operators.current * exact_operators.current * rho)

    state = build_initial_state(problem.system, config.initial_site; tol=1e-14)
    result = run_lattice_frohlich_current_correlation(
        state,
        config,
        problem;
        correlation_time_fs=0.0,
    )
    @test result.times == [0.0]
    @test length(result.correlation) == 1
    @test result.correlation[1] == tt_dot(
        operators.observable,
        operators.left_action * state,
    )

    loose_config = LatticeFrohlichCurrentCorrelationConfig(
        site_count=2,
        site_energies_cm=zeros(2),
        hopping_cm=20.0,
        final_time_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
        state_rounding_tolerance=0.9,
        tamen_tolerance=1e-12,
    )
    loose_problem = lattice_frohlich_current_problem(loose_config)
    loose_operators = build_lattice_frohlich_current_operators(loose_config, loose_problem)
    loose_result = run_lattice_frohlich_current_correlation(
        root_state,
        loose_config,
        loose_problem;
        correlation_time_fs=0.0,
    )
    dense_exact_correlation = tr(
        loose_operators.current * loose_operators.current * rho,
    )
    @test root_density_matrix(loose_result.state, loose_problem.system) ≉
          loose_operators.current * rho
    @test loose_result.correlation[1] ≈ dense_exact_correlation
    @test loose_result.correlation[1] ≉
          tt_dot(loose_operators.observable, loose_result.state)
end

@testset "Lattice Frohlich propagation and progress" begin
    (; config, problem, state) = lattice_frohlich_current_fixture()
    propagated = propagate_lattice_frohlich_fixed_steps(
        state,
        problem,
        config,
        1;
        observe=(step, time, rho) -> (; step, time, rho),
    )
    @test [observation.step for observation in propagated.observations] == [0, 1]
    @test tt_full(propagated.state) ≈ tt_full(state)
    @test_throws ArgumentError propagate_lattice_frohlich_fixed_steps(
        state,
        problem,
        config,
        -1;
        observe=(args...) -> nothing,
    )

    measurement = measure_lattice_frohlich_heom_state(
        state,
        problem.trace_observable,
        problem.population_observables,
    )
    @test measurement.populations ≈ [1.0, 0.0, 0.0]
    @test measurement.trace ≈ 1.0
    normalized = normalize_lattice_frohlich_heom_state(
        (2 - 3im) * state,
        problem.trace_observable,
    )
    @test tt_dot(problem.trace_observable, normalized) ≈ 1
    @test tt_full(normalized) ≈ tt_full(state)

    progress_config = LatticeFrohlichCurrentCorrelationConfig(
        site_count=3,
        site_energies_cm=zeros(3),
        hopping_cm=20.0,
        final_time_fs=5.0,
        time_step_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
        state_rounding_tolerance=1e-14,
        tamen_tolerance=1e-12,
        progress_interval=2,
    )
    equilibration_records = NamedTuple[]
    equilibration = run_lattice_frohlich_equilibration(
        progress_config,
        problem;
        initial_state=state,
        equilibration_time_fs=5.0,
        progress_callback=record -> push!(equilibration_records, record),
    )
    @test [record.step for record in equilibration_records] == [0, 2, 4, 5]
    @test [record.time_fs for record in equilibration_records] == [0.0, 2.0, 4.0, 5.0]
    @test all(record -> isfinite(record.elapsed_seconds) && record.elapsed_seconds >= 0,
              equilibration_records)
    @test [record.maximum_rank for record in equilibration_records] ==
          equilibration.maximum_rank[[1, 3, 5, 6]]
    @test [record.mean_rank for record in equilibration_records] ==
          equilibration.mean_rank[[1, 3, 5, 6]]

    current_records = NamedTuple[]
    current = run_lattice_frohlich_current_correlation(
        state,
        progress_config,
        problem;
        correlation_time_fs=5.0,
        progress_callback=record -> push!(current_records, record),
    )
    @test [record.step for record in current_records] == [0, 2, 4, 5]
    @test [record.time_fs for record in current_records] == [0.0, 2.0, 4.0, 5.0]
    @test all(record -> isfinite(record.elapsed_seconds) && record.elapsed_seconds >= 0,
              current_records)
    @test [record.maximum_rank for record in current_records] ==
          current.maximum_rank[[1, 3, 5, 6]]
    @test [record.mean_rank for record in current_records] ==
          current.mean_rank[[1, 3, 5, 6]]

    physical_config = LatticeFrohlichCurrentCorrelationConfig(
        site_count=2,
        site_energies_cm=[0.0, 15.0],
        hopping_cm=20.0,
        final_time_fs=0.1,
        time_step_fs=0.1,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
        state_rounding_tolerance=1e-14,
        tamen_tolerance=1e-10,
        sweep_count=2,
        local_iterations=2,
        kick_rank=2,
    )
    physical_problem = lattice_frohlich_current_problem(physical_config)
    physical_state = build_initial_state(
        physical_problem.system, physical_config.initial_site; tol=1e-14,
    )
    physical = propagate_lattice_frohlich_fixed_steps(
        physical_state,
        physical_problem,
        physical_config,
        1;
        observe=(step, time, rho) -> (; step, time, norm=norm(tt_full(rho))),
    )
    @test all(observation -> isfinite(observation.norm), physical.observations)
    @test norm(tt_full(physical.state) - tt_full(physical_state)) > 1e-12
end

@testset "Lattice Frohlich reloaded current-correlation outputs" begin
    config = LatticeFrohlichCurrentCorrelationConfig(
        site_count=3,
        site_energies_cm=zeros(3),
        hopping_cm=20.0,
        final_time_fs=1.0,
        time_step_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
        state_rounding_tolerance=1e-14,
        tamen_tolerance=1e-12,
    )
    decomposition = (
        exponents=ComplexF64[0.4],
        coefficients=ComplexF64[0.03 - 0.02im],
    )
    full_problem = build_lattice_frohlich_current_model(config, decomposition)
    state = build_initial_state(full_problem.system, config.initial_site; tol=1e-14)
    zero_liouvillian = TTMatrix([
        zeros(ComplexF64, 1, dimension, dimension, 1)
        for dimension in tt_dims(state)
    ])
    equilibrium_problem = (
        system=full_problem.system,
        liouvillian=zero_liouvillian,
        trace_observable=full_problem.trace_observable,
        population_observables=full_problem.population_observables,
    )
    equilibrium_result = run_lattice_frohlich_equilibration(
        config,
        equilibrium_problem;
        initial_state=state,
        equilibration_time_fs=config.time_step_fs,
    )

    function write_checkpoint(directory)
        return save_lattice_frohlich_equilibration_outputs(
            config,
            decomposition,
            full_problem,
            equilibrium_result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
            plotter=(path, _) -> write(path, "synthetic PNG"),
        )
    end

    @test _lattice_frohlich_current_output_paths("outputs") == (
        state_path="outputs/lattice_frohlich_equilibrium.ttbin",
        metadata_path="outputs/lattice_frohlich_equilibrium_metadata.toml",
        csv_path="outputs/lattice_frohlich_current_correlation.csv",
        png_path="outputs/lattice_frohlich_current_correlation.png",
        rank_png_path="outputs/lattice_frohlich_current_correlation_ranks.png",
    )

    mktempdir() do directory
        checkpoint = write_checkpoint(directory)
        progress_io = IOBuffer()
        reloaded = Ref{Any}(nothing)
        runner = function(loaded_state, runner_config, runner_problem; kwargs...)
            reloaded[] = loaded_state
            return run_lattice_frohlich_current_correlation(
                loaded_state,
                runner_config,
                runner_problem;
                correlation_time_fs=kwargs[:correlation_time_fs],
                progress_callback=kwargs[:progress_callback],
            )
        end
        main_result = lattice_frohlich_current_correlation_main(
            config;
            correlation_time_fs=0.0,
            output_directory=directory,
            decomposition_builder=_ -> decomposition,
            problem_builder=(_, _) -> full_problem,
            correlation_runner=runner,
            plotter=(path, _) -> write(path, "synthetic current PNG"),
            rank_plotter=(path, _) -> write(path, "synthetic rank PNG"),
            progress_io,
        )
        @test tt_full(reloaded[]) ≈ tt_full(load_tt_binary(checkpoint.state_path))
        @test main_result.state_path == checkpoint.state_path
        @test main_result.metadata_path == checkpoint.metadata_path
        @test collect(basename.(values(main_result.outputs))) == [
            "lattice_frohlich_current_correlation.csv",
            "lattice_frohlich_current_correlation.png",
            "lattice_frohlich_current_correlation_ranks.png",
        ]
        @test all(isfile, values(main_result.outputs))
        @test first(split(read(main_result.csv_path, String), '\n')) ==
              "time_fs,correlation_real_fs^-2,correlation_imag_fs^-2,max_rank,mean_rank"
        reloaded_operators = build_lattice_frohlich_current_operators(config, full_problem)
        @test main_result.result.correlation[1] ≈ tt_dot(
            reloaded_operators.observable,
            reloaded_operators.left_action * load_tt_binary(checkpoint.state_path),
        )
        @test occursin("Loaded equilibrium TT ranks", String(take!(progress_io)))
    end

    for missing_path in (:state_path, :metadata_path)
        mktempdir() do directory
            checkpoint = write_checkpoint(directory)
            rm(getproperty(checkpoint, missing_path))
            @test_throws ArgumentError lattice_frohlich_current_correlation_main(
                config;
                correlation_time_fs=0.0,
                output_directory=directory,
                decomposition_builder=_ -> decomposition,
                problem_builder=(_, _) -> full_problem,
            )
        end
    end

    for output_field in (:csv_path, :png_path, :rank_png_path)
        mktempdir() do directory
            write_checkpoint(directory)
            paths = _lattice_frohlich_current_output_paths(directory)
            target = getproperty(paths, output_field)
            write(target, "existing $output_field")
            runner_called = Ref(false)
            @test_throws ArgumentError lattice_frohlich_current_correlation_main(
                config;
                correlation_time_fs=0.0,
                output_directory=directory,
                overwrite=false,
                decomposition_builder=_ -> error("should not decompose"),
                problem_builder=(_, _) -> error("should not build"),
                correlation_runner=(args...; kwargs...) -> (runner_called[] = true),
            )
            @test !runner_called[]
            @test read(target, String) == "existing $output_field"
        end
    end

    mktempdir() do directory
        write_checkpoint(directory)
        paths = _lattice_frohlich_current_output_paths(directory)
        @test_throws ErrorException lattice_frohlich_current_correlation_main(
            config;
            correlation_time_fs=0.0,
            output_directory=directory,
            overwrite=false,
            decomposition_builder=_ -> decomposition,
            problem_builder=(_, _) -> full_problem,
            plotter=(path, _) -> write(path, "synthetic current PNG"),
            rank_plotter=(args...) -> error("forced rank plot failure"),
        )
        @test all(path -> !ispath(path), (paths.csv_path, paths.png_path, paths.rank_png_path))
    end
end

function lattice_frohlich_current_example_snapshot(directory::AbstractString)
    snapshot = String[]
    for (root, directories, files) in walkdir(directory)
        sort!(directories)
        sort!(files)
        for child in directories
            push!(snapshot, "directory:" * relpath(joinpath(root, child), directory))
        end
        for child in files
            path = joinpath(root, child)
            push!(snapshot, "file:" * relpath(path, directory) * ":" * repr(read(path)))
        end
    end
    return snapshot
end

@testset "Lattice Frohlich current standalone environment" begin
    example_directory = joinpath(
        @__DIR__, "..", "examples", "lattice_frohlich_current_correlation",
    )
    project_path = joinpath(example_directory, "Project.toml")
    readme_path = joinpath(example_directory, "README.md")
    expected_project = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
HEOMKit = "e865c079-6a0d-426d-afc2-450809b0c699"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
QFiND = "af16a7c1-792b-4481-9b88-c9c438329a9c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
TTDynamics = "98f59f52-e016-4068-afe5-90126cbc5c1c"
TTSolver = "6fd66e5e-56e0-488c-9f68-e25b72b7e6b0"

[compat]
CairoMakie = "0.12 - 0.15"
HEOMKit = "1"
LinearAlgebra = "1.11"
QFiND = "1"
Statistics = "1.11"
TOML = "1"
TTDynamics = "1"
TTSolver = "1"
julia = "1.11"
"""
    @test isfile(project_path)
    isfile(project_path) && @test read(project_path, String) == expected_project

    ignored_paths = split(read(joinpath(@__DIR__, "..", ".gitignore"), String), '\n')
    @test "/examples/lattice_frohlich_current_correlation/Manifest.toml" in ignored_paths
    @test "/examples/lattice_frohlich_current_correlation/output/" in ignored_paths

    @test isfile(readme_path)
    if isfile(readme_path)
        readme = read(readme_path, String)
        @test occursin("Pkg.develop([Pkg.PackageSpec(path=path) for path in ARGS]); Pkg.instantiate()", readme)
        @test occursin(
            "julia --project=examples/lattice_frohlich_current_correlation \\\n  examples/lattice_frohlich_current_correlation/equilibrate.jl",
            readme,
        )
        @test occursin(
            "julia --project=examples/lattice_frohlich_current_correlation \\\n  examples/lattice_frohlich_current_correlation/current_correlation.jl",
            readme,
        )
        for output in (
            "lattice_frohlich_equilibration.csv",
            "lattice_frohlich_equilibrium.ttbin",
            "lattice_frohlich_equilibrium_metadata.toml",
            "lattice_frohlich_equilibration_populations.png",
            "lattice_frohlich_current_correlation.csv",
            "lattice_frohlich_current_correlation.png",
            "lattice_frohlich_current_correlation_ranks.png",
        )
            @test occursin(output, readme)
        end
        @test occursin("C(t) = Tr[J exp(L t)(J rho_eq)]", readme)
        @test occursin("fs^-2", readme)
        @test occursin("fixed 1000 fs", readme)
        @test occursin("not a state automatically certified to be stationary", readme)
        for parameter in (
            "equilibration time", "Padé order", "TPSD tolerance",
            "hierarchy local size", "time step", "TT truncation/solver tolerances",
        )
            @test occursin(parameter, readme)
        end
        for setting in (
            "site_count = 5",
            "site_energies_cm = zeros(5)",
            "hopping_cm = 400.0",
            "brownian_frequency_cm = 1400.0",
            "brownian_damping_cm = 200.0",
            "reorganization_energy_cm = 600.0",
            "temperature_K = 300.0",
            "initial_site = 1",
            "final_time_fs = 500.0",
            "time_step_fs = 1.0",
            "pade_order = 8",
            "tpsd_tolerance = 2e-2",
            "pade_type = :Nm1",
            "validation_final_time_fs = 100.0",
            "validation_sample_count = 200",
            "bcf_upper_bound_cm = 10_000.0",
            "hierarchy_local_size = 4",
            "temporal_basis_size = 3",
            "tamen_tolerance = 2e-2",
            "operator_tolerance = 1e-10",
            "state_rounding_tolerance = 1e-10",
            "sweep_count = 3",
            "local_iterations = 5",
            "kick_rank = 4",
            "progress_interval = 10",
            "equilibration_time_fs = 1000.0",
            "correlation_time_fs = 200.0",
        )
            @test occursin(setting, readme)
        end
    end
end

@testset "Lattice Frohlich current executable isolation" begin
    example_directory = joinpath(
        @__DIR__, "..", "examples", "lattice_frohlich_current_correlation",
    )
    forbidden_cross_example_include =
        r"include\([^\n]*(?:\.\./holstein|\.\./lattice_frohlich|\.\./holstein_current_correlation)"
    source_files = String[]
    for (root, _, files) in walkdir(example_directory)
        append!(
            source_files,
            joinpath.(root, filter(path -> endswith(path, ".jl"), files)),
        )
    end
    sort!(source_files)
    @test !isempty(source_files)
    for file in source_files
        @test !occursin(forbidden_cross_example_include, read(file, String))
    end

    repository_root = normpath(joinpath(@__DIR__, ".."))
    git_common_directory = readchomp(
        `git -C $repository_root rev-parse --path-format=absolute --git-common-dir`,
    )
    development_root = dirname(dirname(git_common_directory))
    sibling_ttsolver = joinpath(development_root, "TTSolver")
    sibling_heomkit = joinpath(development_root, "HEOMKit")
    sibling_qfind = joinpath(development_root, "QFiND")
    separator = Sys.iswindows() ? ';' : ':'
    load_path = join(
        (repository_root, sibling_ttsolver, sibling_heomkit, sibling_qfind, "@", "@stdlib"),
        separator,
    )
    equilibrate_script = repr(joinpath(example_directory, "equilibrate.jl"))
    correlation_script = repr(joinpath(example_directory, "current_correlation.jl"))
    marker = "lattice-frohlich-current-import-ok"
    snapshot_before = lattice_frohlich_current_example_snapshot(example_directory)
    mktempdir() do fixture_root
        fixture_directory = joinpath(fixture_root, "lattice_frohlich_current_correlation")
        mkpath(fixture_directory)
        for source in (
            "config.jl", "model.jl", "utils.jl", "equilibrate.jl",
            "current_correlation.jl",
        )
            cp(joinpath(example_directory, source), joinpath(fixture_directory, source))
        end
        output_directory = joinpath(fixture_directory, "output")
        mkpath(output_directory)
        write(joinpath(output_directory, "preexisting-user-output.txt"), "preserve me")
        fixture_snapshot = lattice_frohlich_current_example_snapshot(fixture_directory)
        fixture_equilibrate_script = repr(joinpath(fixture_directory, "equilibrate.jl"))
        fixture_correlation_script = repr(joinpath(fixture_directory, "current_correlation.jl"))
        expression =
            "module RepositoryExample; " *
            "include($equilibrate_script); include($correlation_script); " *
            "@assert DEFAULT_LATTICE_FROHLICH_CORRELATION_TIME_FS == 200.0; end; " *
            "module PreexistingOutputFixture; " *
            "include($fixture_equilibrate_script); include($fixture_correlation_script); " *
            "@assert DEFAULT_LATTICE_FROHLICH_CORRELATION_TIME_FS == 200.0; end; " *
            "print($(repr(marker)))"
        command = `$(Base.julia_cmd()) --startup-file=no --compiled-modules=no --project=$example_directory -e $expression`
        command = addenv(
            command, "JULIA_LOAD_PATH" => load_path, "GKSwstype" => "100",
        )
        mktempdir() do working_directory
            stdout_buffer = IOBuffer()
            stderr_buffer = IOBuffer()
            process = run(
                pipeline(
                    Cmd(command; dir=working_directory);
                    stdout=stdout_buffer,
                    stderr=stderr_buffer,
                );
                wait=false,
            )
            wait(process)
            stdout_text = String(take!(stdout_buffer))
            stderr_text = String(take!(stderr_buffer))
            success(process) || println(stderr, stderr_text)
            @test success(process)
            @test stdout_text == marker
            @test isempty(readdir(working_directory))
        end
        @test lattice_frohlich_current_example_snapshot(example_directory) == snapshot_before
        @test lattice_frohlich_current_example_snapshot(fixture_directory) == fixture_snapshot
        @test read(joinpath(output_directory, "preexisting-user-output.txt"), String) ==
              "preserve me"
    end
end
