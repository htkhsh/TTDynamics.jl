using LinearAlgebra
using Test
using TTDynamics
using TTSolver
using HEOMKit

include("../examples/lattice_frohlich/config.jl")
include("../examples/lattice_frohlich/model.jl")
include("../examples/lattice_frohlich/dynamics.jl")
include("../examples/lattice_frohlich/plotting.jl")
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

    two_site_weights = normalized_frohlich_kernel(2)
    @test two_site_weights ≈ [sqrt(8.0) / 3.0 1.0 / 3.0;
                              1.0 / 3.0 sqrt(8.0) / 3.0]
    @test all(isapprox(sum(abs2, two_site_weights[:, n]), 1.0; atol=1e-14) for n in 1:2)
    @test two_site_weights[:, 2] ≈ circshift(two_site_weights[:, 1], 1)

    @test periodic_lattice_distance(1, 3, 4) == 2
    @test periodic_lattice_distance(2, 4, 4) == 2
    even_raw_column = [1.0, 2.0^(-1.5), 5.0^(-1.5), 2.0^(-1.5)]
    even_weights = normalized_frohlich_kernel(4)
    @test even_weights[:, 1] ≈ even_raw_column ./ sqrt(sum(abs2, even_raw_column))
    @test even_weights == transpose(even_weights)
    @test all(isapprox(sum(abs2, even_weights[:, n]), 1.0; atol=1e-14) for n in 1:4)
    @test all(even_weights[:, mod1(n + 1, 4)] == circshift(even_weights[:, n], 1) for n in 1:4)

    even_operators = frohlich_coupling_operators(4)
    @test length(even_operators) == 4
    @test all(isdiag, even_operators)
    @test all(ishermitian, even_operators)
    @test all(diag(even_operators[mod1(m + 1, 4)]) == circshift(diag(even_operators[m]), 1) for m in 1:4)

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
    expected_local_operators = [
        Matrix(Diagonal(ComplexF64.(m == n ? 1.0 : 0.0 for n in 1:5)))
        for m in 1:5
    ]
    @test frohlich_coupling_operators(5; kernel=local_kernel) == expected_local_operators
    @test_throws ArgumentError normalized_frohlich_kernel(1)
    @test_throws ArgumentError normalized_frohlich_kernel(3; kernel=d -> NaN)
    @test_throws ArgumentError normalized_frohlich_kernel(3; kernel=d -> -1.0)
    @test_throws ArgumentError normalized_frohlich_kernel(3; kernel=d -> 0.0)

    kernel_call = Ref(0)
    call_dependent_kernel = _ -> begin
        kernel_call[] += 1
        1.0 + kernel_call[] / 100.0
    end
    @test_throws ErrorException normalized_frohlich_kernel(4; kernel=call_dependent_kernel)
end

@testset "Lattice Frohlich HEOM-TT construction" begin
    config = LatticeFrohlichConfig(
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
    problem = build_lattice_frohlich_model(config, decomposition)
    expected = frohlich_coupling_operators(3)

    @test problem.system.H_sys ≈ periodic_lattice_frohlich_hamiltonian(
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
    local_problem = build_lattice_frohlich_model(config, decomposition; kernel=local_kernel)
    expected_local_couplings = [
        Matrix(Diagonal(ComplexF64.(m == n ? 1.0 : 0.0 for n in 1:3)))
        for m in 1:3
    ]
    @test local_problem.system.noise.V == expected_local_couplings
end

@testset "Lattice Frohlich executable contract" begin
    @test DEFAULT_LATTICE_FROHLICH_CONFIG isa LatticeFrohlichConfig
    previous_config_name = Symbol(join(("Hol", "steinConfig")))
    @test !isdefined(@__MODULE__, previous_config_name) ||
          LatticeFrohlichConfig !== getfield(@__MODULE__, previous_config_name)
    source = read(joinpath(@__DIR__, "..", "examples", "lattice_frohlich",
                           "lattice_frohlich_brownian_heomtt.jl"), String)
    @test !occursin(join(("..", "/holstein")), source)
    @test !occursin(join(("Hol", "steinConfig")), source)
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

    result = (
        times=[0.0, 1.5],
        populations=[1.0 0.25; 0.0 0.75],
        trace=[1.0, 0.99],
        maximum_rank=[2, 3],
        mean_rank=[1.5, 2.0],
    )
    mktempdir() do directory
        path = joinpath(directory, "populations.csv")
        @test write_lattice_frohlich_population_csv(path, result) == path
        @test readlines(path) == [
            "time_fs,population_site_1,population_site_2,trace,max_rank,mean_rank",
            "0.0,1.0,0.0,1.0,2,1.5",
            "1.5,0.25,0.75,0.99,3,2.0",
        ]
    end

    example = abspath(joinpath(
        @__DIR__, "..", "examples", "lattice_frohlich",
        "lattice_frohlich_brownian_heomtt.jl",
    ))
    default_paths = lattice_frohlich_output_paths(dirname(example))
    absent_before = map(path -> !ispath(path), default_paths)
    @test all(absent_before)
    expression = "include($(repr(example))); print(\"lattice-frohlich-import-ok\")"
    command = `$(Base.julia_cmd()) --startup-file=no --compiled-modules=no --project=$(dirname(example)) -e $expression`
    dev_root = dirname(dirname(readchomp(`git rev-parse --path-format=absolute --git-common-dir`)))
    environment_separator = Sys.iswindows() ? ';' : ':'
    example_load_path = join((
        abspath(joinpath(@__DIR__, "..")),
        joinpath(dev_root, "TTSolver"),
        joinpath(dev_root, "HEOMKit"),
        joinpath(dev_root, "QFiND"),
        "@",
        "@stdlib",
    ), environment_separator)
    command = addenv(command,
                     "JULIA_LOAD_PATH" => example_load_path,
                     "GKSwstype" => "100")
    mktempdir() do directory
        output = IOBuffer()
        error_output = IOBuffer()
        process = run(
            pipeline(Cmd(command; dir=directory); stdout=output, stderr=error_output);
            wait=false,
        )
        wait(process)
        text = String(take!(output))
        error_text = String(take!(error_output))
        success(process) || println(stderr, error_text)
        @test success(process)
        @test text == "lattice-frohlich-import-ok"
        @test isempty(readdir(directory))
        absent_after = map(path -> !ispath(path), default_paths)
        @test absent_after == absent_before
        @test all(absent_after)
    end
end
