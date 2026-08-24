active_project = Base.active_project()
if isnothing(active_project) || !isfile(active_project)
    pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "holstein"))
end

using LinearAlgebra
using Statistics
using TTSolver
using TTDynamics
using QFiND
using HEOMKit
using HEOMKit: icm2ifs
using CairoMakie

if !isdefined(@__MODULE__, :HolsteinConfig)
    include("../holstein/utils.jl")
end
if !isdefined(@__MODULE__, :decompose_brownian_bcf)
    include("../holstein/holstein_brownian_heomtt.jl")
end
if !isdefined(@__MODULE__, :normalized_frohlich_kernel)
    include("utils.jl")
end

const DEFAULT_LATTICE_FROHLICH_CONFIG = HolsteinConfig()

function build_lattice_frohlich_heomtt(config::HolsteinConfig, decomposition;
                                        kernel=default_frohlich_kernel)
    validate_config(config)
    H_fs = periodic_holstein_hamiltonian(
        config.site_energies_cm,
        config.hopping_cm,
    ) * icm2ifs
    couplings = frohlich_coupling_operators(config.site_count; kernel)
    baths = [
        BathExp(decomposition.exponents, decomposition.coefficients, coupling)
        for coupling in couplings
    ]
    system = HEOMTTSystem(H_fs, NoiseExp(baths), config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable, population_observables)
end

function lattice_frohlich_output_paths(output_directory::AbstractString)
    return (
        csv=joinpath(output_directory, "lattice_frohlich_brownian_populations.csv"),
        populations=joinpath(output_directory, "lattice_frohlich_brownian_populations.png"),
        trace=joinpath(output_directory, "lattice_frohlich_brownian_trace.png"),
        rank=joinpath(output_directory, "lattice_frohlich_brownian_rank.png"),
    )
end

function save_lattice_frohlich_diagnostic_plots(output_directory, config::HolsteinConfig, result)
    mkpath(output_directory)
    paths = lattice_frohlich_output_paths(output_directory)
    population_figure = Figure(size=(900, 600))
    population_axis = Axis(population_figure[1, 1]; xlabel="Time [fs]",
        ylabel="Population", title="Periodic lattice Frohlich HEOM-TT populations")
    for site in 1:config.site_count
        lines!(population_axis, result.times, result.populations[site, :];
               linewidth=2, label="site $site")
    end
    axislegend(population_axis; position=:rt)
    save(paths.populations, population_figure)

    trace_figure = Figure(size=(700, 450))
    trace_axis = Axis(trace_figure[1, 1]; xlabel="Time [fs]", ylabel="Tr[rho]",
                      title="Lattice Frohlich HEOM-TT trace conservation")
    lines!(trace_axis, result.times, result.trace; linewidth=2, label="trace")
    hlines!(trace_axis, [1.0]; linewidth=1, linestyle=:dash, color=:gray)
    axislegend(trace_axis; position=:rt)
    save(paths.trace, trace_figure)

    rank_figure = Figure(size=(700, 450))
    rank_axis = Axis(rank_figure[1, 1]; xlabel="Time [fs]", ylabel="TT rank",
                     title="Lattice Frohlich HEOM-TT rank evolution")
    lines!(rank_axis, result.times, result.maximum_rank; linewidth=2,
           label="maximum rank")
    lines!(rank_axis, result.times, result.mean_rank; linewidth=2,
           label="mean rank")
    axislegend(rank_axis; position=:rt)
    save(paths.rank, rank_figure)
    return (paths.populations, paths.trace, paths.rank)
end

function lattice_frohlich_main(config=DEFAULT_LATTICE_FROHLICH_CONFIG;
                                output_directory=@__DIR__)
    validate_config(config)
    println("Periodic lattice Frohlich model with normalized long-range diagonal coupling")
    decomposition = decompose_brownian_bcf(config)
    println("  TPSD terms per bath: $(length(decomposition.exponents))")
    println("  TPSD validation relative error: $(decomposition.relative_error)")
    problem = build_lattice_frohlich_heomtt(config, decomposition)
    result = run_dynamics(config, problem)
    paths = lattice_frohlich_output_paths(output_directory)
    mkpath(output_directory)
    write_population_csv(paths.csv, result)
    plot_paths = save_lattice_frohlich_diagnostic_plots(output_directory, config, result)
    println("  Wrote: $(paths.csv)")
    foreach(path -> println("  Wrote: $path"), plot_paths)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    lattice_frohlich_main()
end
