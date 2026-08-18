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
