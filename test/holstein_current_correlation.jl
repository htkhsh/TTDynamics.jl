using LinearAlgebra
using Statistics
using Test
using TTDynamics
using TTSolver
using HEOMKit
using QFiND

include(joinpath(@__DIR__, "..", "examples", "holstein", "config.jl"))
include(joinpath(@__DIR__, "..", "examples", "holstein", "model.jl"))
include(joinpath(@__DIR__, "..", "examples", "holstein", "utils.jl"))

@testset "Holstein compact configuration" begin
    @test DEFAULT_CONFIG isa NamedTuple
    @test DEFAULT_CONFIG.initial_condition == :localized
    @test DEFAULT_CONFIG.site_count == length(DEFAULT_CONFIG.site_energies_cm)
    @test DEFAULT_CONFIG.equilibration_tamen_tolerance == 2e-2
    @test DEFAULT_CONFIG.correlation_tamen_tolerance == 2e-2
    @test !(:tamen_tolerance in propertynames(DEFAULT_CONFIG))
    @test !(:final_time_fs in propertynames(DEFAULT_CONFIG))
    @test !(:validation_final_time_fs in propertynames(DEFAULT_CONFIG))
    @test !(:validation_sample_count in propertynames(DEFAULT_CONFIG))
    @test EQUILIBRATION_TIME_FS > 0
    @test CORRELATION_TIME_FS >= 0
end

@testset "Holstein compact model" begin
    config = merge(DEFAULT_CONFIG, (
        site_count=2,
        site_energies_cm=zeros(2),
        pade_order=1,
        hierarchy_local_size=2,
        operator_tolerance=1e-8,
    ))
    hamiltonian = holstein_hamiltonian(config)
    @test size(hamiltonian) == (2, 2)
    @test ishermitian(hamiltonian)
    @test length(site_projectors(2)) == 2
    @test ishermitian(current_operator(config))
    decomposition = decompose_bath(config)
    @test propertynames(decomposition) == (:exponents, :coefficients)
    problem = build_problem(config, decomposition)
    @test problem.system isa HEOMTTSystem
    @test first(tt_dims(problem.liouvillian)) == heom_tt_dimensions(problem.system)
end

@testset "Holstein propagation helpers" begin
    @test step_count(0.0, 1.0; allow_zero=true) == 0
    @test step_count(3.0, 1.0) == 3
    @test_throws ArgumentError step_count(0.0, 1.0)
    @test_throws ArgumentError step_count(1.1, 1.0)

    config = merge(DEFAULT_CONFIG, (
        site_count=2,
        site_energies_cm=zeros(2),
        pade_order=1,
        hierarchy_local_size=2,
        temporal_basis_size=3,
        equilibration_tamen_tolerance=5e-2,
        correlation_tamen_tolerance=3e-2,
        operator_tolerance=1e-8,
        state_rounding_tolerance=1e-8,
        sweep_count=1,
        local_iterations=2,
    ))
    problem = build_problem(config, decompose_bath(config))
    localized = initial_heom_state(config, problem.system)
    @test tt_dims(localized) == heom_tt_dimensions(problem.system)
    @test isapprox(real(tt_dot(problem.trace_observable, localized)), 1; atol=1e-10)
    gibbs = merge(config, (initial_condition=:gibbs,))
    gibbs_state = initial_heom_state(gibbs, problem.system)
    @test isapprox(real(tt_dot(problem.trace_observable, gibbs_state)), 1; atol=1e-10)

    operators = current_heom_operators(config, problem)
    @test tt_dims(operators.left_action) == tt_dims(problem.liouvillian)
    @test tt_dims(operators.observable) == heom_tt_dimensions(problem.system)
    workspace = TAMEnWorkspace(ComplexF64)
    stepped = tamen_step(localized, problem.liouvillian, config,
                         problem.trace_observable, workspace)
    @test tt_dims(stepped) == tt_dims(localized)
    measured = measure_density(stepped, problem)
    @test length(measured.populations) == config.site_count
    @test length(measured.coherence_pairs) == 1
    @test length(measured.coherences) == 1
    @test isfinite(measured.trace)
    @test measured.maximum_rank >= 1

    equilibration_progress = NamedTuple[]
    equilibration = run_equilibration(
        config,
        problem;
        duration_fs=1.0,
        progress_callback=progress -> push!(equilibration_progress, progress),
    )
    @test equilibration.times == [0.0, 1.0]
    @test size(equilibration.populations) == (2, 2)
    @test size(equilibration.coherences) == (1, 2)
    @test length(equilibration.trace) == 2
    @test [progress.step for progress in equilibration_progress] == [0, 1]

    correlation_progress = NamedTuple[]
    correlation = run_current_correlation(
        localized,
        config,
        problem;
        duration_fs=0.0,
        progress_callback=progress -> push!(correlation_progress, progress),
    )
    @test correlation.times == [0.0]
    @test length(correlation.correlation) == 1
    @test [progress.step for progress in correlation_progress] == [0]
end

@testset "Holstein example persistence" begin
    config = merge(DEFAULT_CONFIG, (site_count=2, site_energies_cm=zeros(2)))
    decomposition = (exponents=ComplexF64[1 + 0im], coefficients=ComplexF64[1 + 0im])
    @test isdefined(Main, :write_equilibrium_metadata)
    @test isdefined(Main, :write_equilibration_csv)
    mktempdir() do directory
        metadata_path = joinpath(directory, "state.toml")
        csv_path = joinpath(directory, "equilibration.csv")
        fake_problem = (system=(nb=[2, 2],),)
        fake_state = (cores=Any[],)
        metadata = Dict("identifier" => "TTDynamics Holstein HEOM-TT equilibrium",
                        "version" => 1,
                        "site_count" => 2,
                        "initial_condition" => "localized",
                        "equilibration_time_fs" => 1.0,
                        "hierarchy_sizes" => [2, 2],
                        "tt_dimensions" => [2, 2, 2, 2])
        write_equilibrium_metadata(metadata_path, metadata)
        @test read_equilibrium_metadata(metadata_path)["site_count"] == 2
        result = (; times=[0.0], populations=[1.0 0.0;],
                  coherence_pairs=Tuple{Int,Int}[], coherences=zeros(ComplexF64, 0, 1),
                  trace=[1.0], maximum_rank=[1], mean_rank=[1.0])
        write_equilibration_csv(csv_path, result)
        @test read(csv_path, String) |> contains("population_site_1")
        @test decomposition.exponents[1] == 1 + 0im
        @test fake_problem.system.nb == [2, 2]
        @test fake_state.cores == Any[]
    end
end
