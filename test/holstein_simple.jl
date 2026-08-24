using Test

@testset "Single-file Holstein HEOM-TT example" begin
    example_directory = joinpath(@__DIR__, "..", "examples", "holstein-simple")
    example_path = joinpath(example_directory, "holstein_brownian_heomtt.jl")
    entries_before = isdir(example_directory) ? Set(readdir(example_directory)) : Set{String}()

    example_module = Module(:HolsteinSimpleExampleTest)
    Base.include(example_module, example_path)

    entries_after = Set(readdir(example_directory))
    @test entries_after == entries_before
    @test isdefined(example_module, :main)
    @test isdefined(example_module, :simple_holstein_hamiltonian)
    @test isdefined(example_module, :simple_holstein_output_paths)

    hamiltonian = example_module.simple_holstein_hamiltonian(
        [10.0, 20.0, 30.0],
        4.0,
    )
    @test hamiltonian == ComplexF64[
        10 -4 -4
        -4 20 -4
        -4 -4 30
    ]

    paths = example_module.simple_holstein_output_paths("result")
    @test paths == (
        csv=joinpath("result", "holstein_brownian_populations.csv"),
        populations=joinpath("result", "holstein_brownian_populations.png"),
        trace=joinpath("result", "holstein_brownian_trace.png"),
        rank=joinpath("result", "holstein_brownian_rank.png"),
    )

    mktempdir() do directory
        result = example_module.main(
            site_count=2,
            site_energies_cm=zeros(2),
            hopping_cm=20.0,
            brownian_frequency_cm=200.0,
            brownian_damping_cm=50.0,
            reorganization_energy_cm=5.0,
            final_time_fs=1.0,
            time_step_fs=1.0,
            pade_order=2,
            tpsd_tolerance=1e-1,
            validation_final_time_fs=5.0,
            validation_sample_count=10,
            hierarchy_local_size=2,
            tamen_tolerance=1e-1,
            operator_tolerance=1e-10,
            state_rounding_tolerance=1e-8,
            sweep_count=1,
            local_iterations=1,
            kick_rank=1,
            progress_interval=1,
            output_directory=directory,
        )
        @test result.times == [0.0, 1.0]
        @test size(result.populations) == (2, 2)
        @test length(result.trace) == 2
        @test all(isfile, values(result.paths))
    end
end
