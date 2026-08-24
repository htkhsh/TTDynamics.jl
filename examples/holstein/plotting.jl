holstein_output_paths(directory::AbstractString) = (
    csv=joinpath(directory, "holstein_brownian_populations.csv"),
    populations=joinpath(directory, "holstein_brownian_populations.png"),
    trace=joinpath(directory, "holstein_brownian_trace.png"),
    rank=joinpath(directory, "holstein_brownian_rank.png"),
)

"""
    save_holstein_results(output_directory, config, result)

Save population CSV, trace-conservation, and TT-rank diagnostics in
`output_directory`.
"""
function save_holstein_results(output_directory, config::HolsteinConfig, result)
    mkpath(output_directory)
    pole_decay_cm = config.brownian_damping_cm / 2
    paths = holstein_output_paths(output_directory)

    write_holstein_population_csv(paths.csv, result)
    population_figure = Figure(size=(900, 600))
    population_axis = Axis(
        population_figure[1, 1];
        xlabel="Time [fs]",
        ylabel="Population",
        title="Periodic Holstein HEOM-TT populations\n" *
              "N=$(config.site_count), J=$(config.hopping_cm) cm⁻¹\n" *
              "Ω=$(config.brownian_frequency_cm) cm⁻¹, " *
              "QFiND Γ_Q=$(config.brownian_damping_cm) cm⁻¹ " *
              "(pole decay Γ_Q/2=$pole_decay_cm cm⁻¹), " *
              "λ=$(config.reorganization_energy_cm) cm⁻¹, T=$(config.temperature_K) K",
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
        ylabel="Tr[ρ]",
        title="HEOM-TT trace conservation",
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
        title="HEOM-TT rank evolution",
    )
    lines!(
        rank_axis,
        result.times,
        result.maximum_rank;
        linewidth=2,
        label="maximum rank",
    )
    lines!(
        rank_axis,
        result.times,
        result.mean_rank;
        linewidth=2,
        label="mean rank",
    )
    axislegend(rank_axis; position=:rt)
    save(paths.rank, rank_figure)

    return (; csv_path=paths.csv, plot_paths=(paths.populations, paths.trace, paths.rank))
end
