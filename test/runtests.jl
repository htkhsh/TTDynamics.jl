using TTDynamics
using TTSolver
using Test
using HEOMKit

@testset "TTDynamics.jl" begin
    # Write your tests here.
end

include("tt_io.jl")
include("holstein_current_correlation.jl")
include("heom_twin_space.jl")
include("lattice_frohlich.jl")
include("lattice_frohlich_current_correlation.jl")
include("holstein_simple.jl")

include("../examples/tfd/ksl/utils.jl")
include("../examples/holstein/config.jl")
include("../examples/holstein/model.jl")
include("../examples/holstein/dynamics.jl")
include("../examples/holstein/plotting.jl")

@testset "Holstein-family examples are self-contained" begin
    roots = [
        joinpath(@__DIR__, "..", "examples", "holstein"),
        joinpath(@__DIR__, "..", "examples", "lattice_frohlich"),
        joinpath(@__DIR__, "..", "examples", "holstein_current_correlation"),
        joinpath(@__DIR__, "..", "examples", "lattice_frohlich_current_correlation"),
    ]
    cross_example_include = Regex(
        raw"include\([^\n]*(?:" *
        raw"(?:\.\./|examples/)(?:holstein|lattice_frohlich|holstein_current_correlation)(?:/|\")|" *
        raw"joinpath\([^)]*\"\.\.\"[^)]*\"(?:holstein|lattice_frohlich|holstein_current_correlation)\")",
    )

    cross_holstein_path = join(("..", "/holstein"))
    @test occursin(cross_example_include, "include(\"$cross_holstein_path/config.jl\")")
    @test occursin(
        cross_example_include,
        "include(joinpath(@__DIR__, \"..\", \"lattice_frohlich\", \"config.jl\"))",
    )
    @test !occursin(
        cross_example_include,
        "include(\"lattice_frohlich_brownian_heomtt.jl\")",
    )

    for root in roots
        for file in filter(path -> endswith(path, ".jl"), readdir(root; join=true))
            source = read(file, String)
            @test !occursin(cross_example_include, source)
        end
    end
    @test !isfile(joinpath(roots[1], "utils.jl"))
    @test !isfile(joinpath(roots[2], "utils.jl"))
    @test HolsteinConfig !== LatticeFrohlichConfig
    @test HolsteinConfig !== HolsteinCurrentCorrelationConfig
    @test LatticeFrohlichConfig !== HolsteinCurrentCorrelationConfig
    @test LatticeFrohlichCurrentCorrelationConfig !== HolsteinConfig
    @test LatticeFrohlichCurrentCorrelationConfig !== LatticeFrohlichConfig
    @test LatticeFrohlichCurrentCorrelationConfig !== HolsteinCurrentCorrelationConfig
end

function assert_holstein_family_default_config(config)
    expected = (
        site_count=5,
        site_energies_cm=zeros(5),
        hopping_cm=400.0,
        brownian_frequency_cm=1400.0,
        brownian_damping_cm=200.0,
        reorganization_energy_cm=600.0,
        temperature_K=300.0,
        initial_site=1,
        final_time_fs=500.0,
        time_step_fs=1.0,
        pade_order=8,
        tpsd_tolerance=2e-2,
        pade_type=:Nm1,
        validation_final_time_fs=100.0,
        validation_sample_count=200,
        bcf_upper_bound_cm=10_000.0,
        hierarchy_local_size=4,
        temporal_basis_size=3,
        tamen_tolerance=2e-2,
        operator_tolerance=1e-10,
        state_rounding_tolerance=1e-10,
        sweep_count=3,
        local_iterations=5,
        kick_rank=4,
        progress_interval=10,
    )
    @test fieldnames(typeof(config)) == keys(expected)
    for field in keys(expected)
        @test getproperty(config, field) == getproperty(expected, field)
    end
end

@testset "Holstein-family configuration defaults" begin
    @testset "HolsteinConfig" begin
        assert_holstein_family_default_config(HolsteinConfig())
    end
    @testset "LatticeFrohlichConfig" begin
        assert_holstein_family_default_config(LatticeFrohlichConfig())
    end
    @testset "HolsteinCurrentCorrelationConfig" begin
        assert_holstein_family_default_config(HolsteinCurrentCorrelationConfig())
    end
end

@testset "Periodic Holstein example layout" begin
    config = HolsteinConfig()
    @test config.final_time_fs == 500.0
    @test config.tpsd_tolerance == 2e-2
    @test config.validation_sample_count == 200
    @test validate_holstein_config(config) === config
    @test length(holstein_site_projectors(config.site_count)) == config.site_count
    mktempdir() do directory
        paths = holstein_output_paths(directory)
        @test paths == (
            csv=joinpath(directory, "holstein_brownian_populations.csv"),
            populations=joinpath(directory, "holstein_brownian_populations.png"),
            trace=joinpath(directory, "holstein_brownian_trace.png"),
            rank=joinpath(directory, "holstein_brownian_rank.png"),
        )
        @test all(path -> !ispath(path), paths)
    end

    result = (
        times=[0.0, 1.5],
        populations=[1.0 0.25; 0.0 0.75],
        trace=[1.0, 0.99],
        maximum_rank=[2, 3],
        mean_rank=[1.5, 2.0],
    )
    mktempdir() do directory
        path = joinpath(directory, "populations.csv")
        @test write_holstein_population_csv(path, result) == path
        @test readlines(path) == [
            "time_fs,population_site_1,population_site_2,trace,max_rank,mean_rank",
            "0.0,1.0,0.0,1.0,2,1.5",
            "1.5,0.25,0.75,0.99,3,2.0",
        ]
    end
end

@testset "Periodic Holstein model utilities" begin
    H = periodic_holstein_hamiltonian(zeros(5), 400.0)
    @test ishermitian(H)
    @test diag(H) == zeros(5)
    @test H[1, 2] == H[2, 3] == H[3, 4] == H[4, 5] == H[5, 1] == -400.0
    @test H[1, 3] == H[2, 4] == H[3, 5] == 0.0

    H2 = periodic_holstein_hamiltonian([10.0, 20.0], 3.0)
    @test H2 == ComplexF64[10 -3; -3 20]

    projectors = holstein_site_projectors(5)
    @test all(ishermitian, projectors)
    @test sum(projectors) == Matrix{ComplexF64}(I, 5, 5)
    @test all(iszero(projectors[i] * projectors[j]) for i in 1:5 for j in 1:5 if i != j)

    config = HolsteinConfig()
    @test validate_holstein_config(config) === config
    @test config.site_count == 5
    @test config.hopping_cm == 400.0
    @test config.brownian_frequency_cm == 1400.0
    @test config.reorganization_energy_cm == 600.0
    @test config.brownian_damping_cm == 200.0
    @test config.temperature_K == 300.0
    @test config.temporal_basis_size == 3
    @test config.pade_order == 8
    @test config.tpsd_tolerance == 2e-2
    @test config.pade_type == :Nm1
    @test config.validation_final_time_fs == 100.0
    @test config.validation_sample_count == 200
    @test_throws ArgumentError HolsteinConfig(site_count=1)
    @test_throws ArgumentError HolsteinConfig(initial_site=6)
    @test_throws ArgumentError HolsteinConfig(time_step_fs=3.0, final_time_fs=100.0)
    @test_throws ArgumentError HolsteinConfig(hierarchy_local_size=0)
    @test_throws ArgumentError HolsteinConfig(temporal_basis_size=1)
    @test_throws ArgumentError HolsteinConfig(temporal_basis_size=2)
    @test_throws ArgumentError HolsteinConfig(temporal_basis_size=4)
    @test_throws ArgumentError HolsteinConfig(pade_order=0)
    @test_throws ArgumentError HolsteinConfig(tpsd_tolerance=0.0)
    @test_throws ArgumentError HolsteinConfig(pade_type=:invalid)
    @test_throws ArgumentError HolsteinConfig(validation_sample_count=1)
    @test_throws ArgumentError HolsteinConfig(
        brownian_frequency_cm=100.0,
        brownian_damping_cm=200.0,
    )
end

@testset "Periodic Holstein HEOM-TT construction" begin
    config = HolsteinConfig(
        site_count=3,
        site_energies_cm=[0.0, 10.0, -5.0],
        hierarchy_local_size=2,
        final_time_fs=1.0,
        time_step_fs=1.0,
        operator_tolerance=1e-14,
    )
    decomposition = (
        exponents=ComplexF64[0.25],
        coefficients=ComplexF64[0.03 - 0.01im],
    )
    problem = build_holstein_model(config, decomposition)

    @test problem.system.H_sys ≈ periodic_holstein_hamiltonian(
        config.site_energies_cm,
        config.hopping_cm,
    ) * icm2ifs
    @test heom_tt_dimensions(problem.system) == [3, 3, 2, 2, 2]
    @test length(problem.population_observables) == 3
    initial = build_initial_state(problem.system, config.initial_site; tol=1e-14)
    @test tt_dims(initial) == [3, 3, 2, 2, 2]
    @test isapprox(real(tt_dot(problem.trace_observable, initial)), 1.0; atol=1e-9)
end

@testset "KSL example utilities" begin
    @test admissible_tt_ranks([2, 3, 4]) == [1, 2, 4, 1]
    @test admissible_tt_ranks([2, 3, 4], 3) == [1, 2, 3, 1]
    @test_throws ArgumentError admissible_tt_ranks(Int[], 30)
    @test_throws ArgumentError admissible_tt_ranks([2, 0, 4], 30)
    @test_throws ArgumentError admissible_tt_ranks([2, 3, 4], 0)

    x0 = initial_tt_state([SpinSpace(0.5), BosonSpace(3), BosonSpace(4)];
                          state_index=1, site=1)
    target_ranks = admissible_tt_ranks(tt_dims(x0), 30)
    padded = pad_tt_ranks(x0, target_ranks)

    @test tt_ranks(padded) == target_ranks
    @test tt_full(padded) == tt_full(x0)
    @test_throws ArgumentError pad_tt_ranks(x0, [1, 2, 1])
    @test_throws ArgumentError pad_tt_ranks(x0, [2, 2, 4, 1])
    @test_throws ArgumentError pad_tt_ranks(padded, [1, 1, 1, 1])

    spin_superposition = product_tt_state([
        ComplexF64[1 / sqrt(2), 1im / sqrt(2)],
        ComplexF64[1, 0, 0],
    ])
    @test first_site_populations(spin_superposition) ≈ [0.5, 0.5]
end

@testset "KSL entangled first-site populations" begin
    # This catches an implementation that drops complex off-diagonal bond
    # environments while contracting the bath sites.
    cores = [
        reshape(ComplexF64[1 + 2im, 2 - im, -1im, 1 - 3im], 1, 2, 2),
        reshape(
            ComplexF64[
                1, 1im, 2 - im, -1,
                2im, 1 - im, -2, 1 + 2im,
            ],
            2,
            2,
            2,
        ),
        reshape(ComplexF64[1 - im, 2im, -1, 1 + im, 2 - 2im, -2im], 2, 3, 1),
    ]
    psi = TTTensor(cores)
    dense_psi = tt_full(psi)
    # tt_full returns kron-compatible axes, so the first TT core is the final
    # dense axis.  This dense expansion is independent of the observable
    # contraction below.
    dense_populations = [sum(abs2, @view dense_psi[:, :, state]) for state in 1:2]

    @test first_site_populations(psi) ≈ dense_populations
end

@testset "KSL padded-state evolution" begin
    sigma_x = ComplexF64[0 1; 1 0]
    number = ComplexF64[0 0; 0 1]
    identity2 = Matrix{ComplexF64}(I, 2, 2)
    H = tt_mkron(sigma_x, identity2) + 0.25 * tt_mkron(identity2, number)
    x0 = product_tt_state([
        ComplexF64[1, 0],
        ComplexF64[1, 0],
    ])
    ranks = admissible_tt_ranks(tt_dims(x0), 30)
    x0 = pad_tt_ranks(x0, ranks)
    x1 = tt_ksl(x0, 1im * H, 0.1; symmetric=true, rmax=30)

    @test norm(x1) ≈ norm(x0) rtol=1e-11
end

@testset "Holstein multi-bath HEOM-TT initial state" begin
    projectors = holstein_site_projectors(2)
    baths = [BathExp(ComplexF64[0.1], ComplexF64[0.02], V) for V in projectors]
    noise = NoiseExp(baths)
    system = HEOMTTSystem(ComplexF64[0 -1; -1 0], noise, 2)
    liouvillian, trace_observable, populations = build_heom_liouvillian(system; tol=1e-12)
    rho0 = build_initial_state(system, 1; tol=1e-12)

    @test tt_dims(liouvillian) == ([2, 2, 2, 2], [2, 2, 2, 2])
    @test tt_dims(trace_observable) == [2, 2, 2, 2]
    @test all(population -> tt_dims(population) == [2, 2, 2, 2], populations)
    @test tt_dims(rho0) == [2, 2, 2, 2]
    @test real(tt_dot(trace_observable, rho0)) ≈ 1.0
    @test real(tt_dot(populations[1], rho0)) ≈ 1.0
    @test real(tt_dot(populations[2], rho0)) ≈ 0.0 atol=1e-14
end
