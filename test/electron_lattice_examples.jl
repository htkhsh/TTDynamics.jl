using LinearAlgebra
using Test
using TTDynamics
using TTSolver

module HolsteinElectronLatticeExample
include(joinpath(@__DIR__, "..", "examples", "holstein", "equilibrate.jl"))
end

module FrohlichElectronLatticeExample
include(joinpath(@__DIR__, "..", "examples", "lattice_frohlich", "equilibrate.jl"))
end

@testset "Electron–lattice measurements in both equilibration examples" begin
    for example in (HolsteinElectronLatticeExample, FrohlichElectronLatticeExample)
        @testset "$(nameof(example))" begin
            config = merge(example.DEFAULT_CONFIG, (
                site_count=3, site_energies_cm=zeros(3), hopping_cm=500.0,
                initial_condition=:gibbs,
                hierarchy_local_size=2, time_step_fs=0.02,
                equilibration_tamen_tolerance=1e-10, temporal_basis_size=3,
                operator_tolerance=1e-13, sweep_count=12, local_iterations=12,
                blas_threads=1,
            ))
            example.configure_blas(config)
            decomposition = (; exponents=ComplexF64[0.1], coefficients=ComplexF64[0.05 - 0.01im])
            problem = example.build_problem(config, decomposition)
            initial = example.initial_heom_state(config, problem.system)
            observation = example.measure_density(initial, problem)
            @test hasproperty(observation, :electron_lattice)
            if hasproperty(observation, :electron_lattice)
                @test observation.electron_lattice.correlation ≈ zeros(3)
                result = example.run_equilibration(config, problem; duration_fs=0.04)
                @test result.lattice_distances == [0, 1, 2]
                @test size(result.electron_lattice_correlation) == (3, 3)
                @test size(result.bath_mean) == (3, 3)
                @test result.electron_lattice_correlation[:, 1] ≈ zeros(3)
                @test maximum(abs, result.electron_lattice_correlation[:, end]) > 1e-7

                # Independent dense hierarchy evolution and root observations.
                dense = exp(0.04 * tt_full(problem.liouvillian)) * vec(tt_full(initial))
                trace = dot(vec(tt_full(problem.trace_observable)), dense)
                p = [dot(vec(tt_full(o)), dense) / trace for o in problem.population_observables]
                joint = zeros(ComplexF64, 3, 3)
                for l in 1:3
                    bath_action = build_heom_bath_coordinate_left_action(problem.system, l)
                    bath_state = tt_full(bath_action) * dense
                    for j in 1:3
                        joint[j, l] = dot(vec(tt_full(problem.population_observables[j])), bath_state) / trace
                    end
                end
                means = vec(sum(joint; dims=1))
                expected = [sum(joint[j, mod1(j+r, 3)] - p[j] * means[mod1(j+r, 3)]
                                for j in 1:3) for r in 0:2]
                @test result.electron_lattice_correlation[:, end] ≈ expected atol=1e-7
                @test result.bath_mean[:, end] ≈ means atol=1e-7
                @test sum(result.electron_lattice_correlation[:, end]) ≈ 0 atol=1e-12
                mktempdir() do directory
                    csv = joinpath(directory, "correlation.csv")
                    example.write_electron_lattice_correlation_csv(csv, result)
                    rows = split.(readlines(csv), ',')
                    @test length(rows) == 10
                    @test rows[1][1:2] == ["time_fs", "distance"]
                    @test parse.(Int, getindex.(rows[2:4], 2)) == [0, 1, 2]
                    @test parse(Float64, rows[end][3]) ≈ real(expected[end]) atol=1e-7
                    @test_throws ArgumentError example.write_electron_lattice_correlation_csv(csv, result; overwrite=false)
                    # Plot smoke check includes the initial and evolved slices.
                    Base.include(example, joinpath(dirname(pathof(TTDynamics)), "..", "examples",
                        example === HolsteinElectronLatticeExample ? "holstein" : "lattice_frohlich", "plotting.jl"))
                    plot = joinpath(directory, "correlation.png")
                    plotter = Base.invokelatest(getfield, example, :save_electron_lattice_plot)
                    Base.invokelatest(plotter, plot, result)
                    @test filesize(plot) > 1000
                end
            end
        end
    end
end
