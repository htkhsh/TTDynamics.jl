using LinearAlgebra
using Test
using TTDynamics
using TTSolver
using HEOMKit

include("../examples/lattice_frohlich_current_correlation/config.jl")
include("../examples/lattice_frohlich_current_correlation/model.jl")
include("../examples/lattice_frohlich_current_correlation/utils.jl")

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
    @test_throws ArgumentError LatticeFrohlichCurrentCorrelationConfig(site_count=1)
    @test_throws ArgumentError LatticeFrohlichCurrentCorrelationConfig(temporal_basis_size=4)
end

@testset "Lattice Frohlich current model" begin
    @test periodic_lattice_frohlich_distance(1, 3, 4) == 2
    @test periodic_lattice_frohlich_distance(1, 4, 4) == 1
    weights = normalized_lattice_frohlich_current_kernel(4)
    @test size(weights) == (4, 4)
    @test all(n -> sum(abs2, weights[:, n]) ≈ 1, 1:4)
    @test weights ≈ transpose(weights)
    @test weights[:, 2] ≈ circshift(weights[:, 1], 1)
    operators = lattice_frohlich_current_coupling_operators(4)
    @test all(isdiag, operators)
    @test all(ishermitian, operators)
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

    problem = lattice_frohlich_current_problem(config)
    operators = build_lattice_frohlich_current_operators(config, problem)
    @test tt_dims(operators.left_action) == tt_dims(problem.liouvillian)
    @test tt_dims(operators.observable) == heom_tt_dimensions(problem.system)

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
end
