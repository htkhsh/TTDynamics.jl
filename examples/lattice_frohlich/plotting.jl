using CairoMakie

lattice_frohlich_output_paths(directory::AbstractString) = (
    csv=joinpath(directory, "lattice_frohlich_brownian_populations.csv"),
    populations=joinpath(directory, "lattice_frohlich_brownian_populations.png"),
    trace=joinpath(directory, "lattice_frohlich_brownian_trace.png"),
    rank=joinpath(directory, "lattice_frohlich_brownian_rank.png"),
)

"""
    save_lattice_frohlich_results(output_directory, config, result)

Save population CSV, trace-conservation, and TT-rank diagnostics in
`output_directory`.
"""
function save_lattice_frohlich_results(output_directory, config::LatticeFrohlichConfig, result)
    mkpath(output_directory)
    paths = lattice_frohlich_output_paths(output_directory)

    write_lattice_frohlich_population_csv(paths.csv, result)
    population_figure = Figure(size=(900, 600))
    population_axis = Axis(
        population_figure[1, 1];
        xlabel="Time [fs]",
        ylabel="Population",
        title="Periodic lattice Frohlich HEOM-TT populations",
    )
    for site in 1:config.site_count
        lines!(
            population_axis,
            result.times,
            result.populations[site, :];
            linewidth=2,
            label="site $site",
        )
    end
    axislegend(population_axis; position=:rt)
    save(paths.populations, population_figure)

    trace_figure = Figure(size=(700, 450))
    trace_axis = Axis(
        trace_figure[1, 1];
        xlabel="Time [fs]",
        ylabel="Tr[rho]",
        title="Lattice Frohlich HEOM-TT trace conservation",
    )
    lines!(trace_axis, result.times, result.trace; linewidth=2, label="trace")
    hlines!(trace_axis, [1.0]; linewidth=1, linestyle=:dash, color=:gray)
    axislegend(trace_axis; position=:rt)
    save(paths.trace, trace_figure)

    rank_figure = Figure(size=(700, 450))
    rank_axis = Axis(
        rank_figure[1, 1];
        xlabel="Time [fs]",
        ylabel="TT rank",
        title="Lattice Frohlich HEOM-TT rank evolution",
    )
    lines!(rank_axis, result.times, result.maximum_rank; linewidth=2, label="maximum rank")
    lines!(rank_axis, result.times, result.mean_rank; linewidth=2, label="mean rank")
    axislegend(rank_axis; position=:rt)
    save(paths.rank, rank_figure)

    return (; csv_path=paths.csv, plot_paths=(paths.populations, paths.trace, paths.rank))
end
