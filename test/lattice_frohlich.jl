using LinearAlgebra
using Test
using TTDynamics
using TTSolver
using HEOMKit

include("../examples/holstein/utils.jl")
include("../examples/lattice_frohlich/utils.jl")
include("../examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl")

@testset "Lattice Frohlich kernel utilities" begin
    @test periodic_lattice_distance(1, 1, 5) == 0
    @test periodic_lattice_distance(1, 5, 5) == 1
    @test periodic_lattice_distance(2, 5, 6) == 3
    @test periodic_lattice_distance(5, 1, 5) == 1
    @test_throws ArgumentError periodic_lattice_distance(0, 1, 5)
    @test_throws ArgumentError periodic_lattice_distance(1, 6, 5)
    @test_throws ArgumentError periodic_lattice_distance(1, 1, 1)

    @test default_frohlich_kernel(0) == 1.0
    @test default_frohlich_kernel(1) == 2.0^(-1.5)
    @test default_frohlich_kernel(2) == 5.0^(-1.5)
    @test_throws ArgumentError default_frohlich_kernel(-1)

    weights = normalized_frohlich_kernel(5)
    @test size(weights) == (5, 5)
    @test all(isfinite, weights)
    @test all(>(0), weights)
    @test weights == transpose(weights)
    @test all(isapprox(sum(abs2, weights[:, n]), 1.0; atol=1e-14) for n in 1:5)
    @test all(weights[:, mod1(n + 1, 5)] == circshift(weights[:, n], 1) for n in 1:5)

    operators = frohlich_coupling_operators(5)
    @test length(operators) == 5
    @test all(ishermitian, operators)
    @test all(isdiag, operators)
    @test all(diag(operators[m]) == weights[m, :] for m in 1:5)

    local_kernel = d -> d == 0 ? 1.0 : 0.0
    @test frohlich_coupling_operators(5; kernel=local_kernel) == site_projectors(5)
    @test_throws ArgumentError normalized_frohlich_kernel(1)
    @test_throws ArgumentError normalized_frohlich_kernel(3; kernel=d -> NaN)
    @test_throws ArgumentError normalized_frohlich_kernel(3; kernel=d -> -1.0)
    @test_throws ArgumentError normalized_frohlich_kernel(3; kernel=d -> 0.0)
end

@testset "Lattice Frohlich HEOM-TT construction" begin
    config = HolsteinConfig(
        site_count=3,
        site_energies_cm=[0.0, 10.0, -5.0],
        hierarchy_local_size=2,
        final_time_fs=1.0,
        time_step_fs=1.0,
    )
    decomposition = (
        exponents=ComplexF64[0.25],
        coefficients=ComplexF64[0.03 - 0.01im],
    )
    problem = build_lattice_frohlich_heomtt(config, decomposition)
    expected = frohlich_coupling_operators(3)

    @test problem.system.H_sys ≈ periodic_holstein_hamiltonian(
        config.site_energies_cm,
        config.hopping_cm,
    ) * icm2ifs
    @test problem.system.noise.V == expected
    @test heom_tt_dimensions(problem.system) == [3, 3, 2, 2, 2]
    @test length(problem.population_observables) == 3
    initial = build_initial_state(problem.system, config.initial_site; tol=1e-12)
    @test tt_dims(initial) == [3, 3, 2, 2, 2]
    @test isapprox(real(tt_dot(problem.trace_observable, initial)), 1.0; atol=1e-9)

    local_kernel = d -> d == 0 ? 1.0 : 0.0
    local_problem = build_lattice_frohlich_heomtt(config, decomposition; kernel=local_kernel)
    @test local_problem.system.noise.V == site_projectors(3)
end

@testset "Lattice Frohlich executable contract" begin
    @test DEFAULT_LATTICE_FROHLICH_CONFIG isa HolsteinConfig
    project_text = read(joinpath(@__DIR__, "..", "examples", "lattice_frohlich", "Project.toml"), String)
    @test occursin("HEOMKit = \"e865c079-6a0d-426d-afc2-450809b0c699\"", project_text)
    @test occursin("TTDynamics = \"98f59f52-e016-4068-afe5-90126cbc5c1c\"", project_text)
    @test occursin("QFiND = \"af16a7c1-792b-4481-9b88-c9c438329a9c\"", project_text)
    @test lattice_frohlich_output_paths("/tmp/example") == (
        csv=joinpath("/tmp/example", "lattice_frohlich_brownian_populations.csv"),
        populations=joinpath("/tmp/example", "lattice_frohlich_brownian_populations.png"),
        trace=joinpath("/tmp/example", "lattice_frohlich_brownian_trace.png"),
        rank=joinpath("/tmp/example", "lattice_frohlich_brownian_rank.png"),
    )

    example = abspath(joinpath(
        @__DIR__, "..", "examples", "lattice_frohlich",
        "lattice_frohlich_brownian_heomtt.jl",
    ))
    expression = "include($(repr(example))); print(\"lattice-frohlich-import-ok\")"
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(example)) -e $expression`
    mktempdir() do directory
        output = IOBuffer()
        process = run(pipeline(Cmd(command; dir=directory); stdout=output, stderr=output); wait=false)
        wait(process)
        text = String(take!(output))
        success(process) || println(stderr, text)
        @test success(process)
        @test occursin("lattice-frohlich-import-ok", text)
        @test isempty(readdir(directory))
    end
end
