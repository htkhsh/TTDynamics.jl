using CairoMakie

function _save_equilibration_population_plot(path::AbstractString, result)::String
    length(result.times) == size(result.populations, 2) ||
        throw(ArgumentError("population column count must equal times length"))
    figure = Figure(size=(900, 600))
    axis = Axis(
        figure[1, 1];
        xlabel="Time (fs)",
        ylabel="Population",
        title="Holstein HEOM-TT equilibration populations",
    )
    for site in axes(result.populations, 1)
        lines!(axis, result.times, result.populations[site, :];
               linewidth=2, label="Site $site")
    end
    axislegend(axis; position=:rt)
    save(String(path), figure)
    return String(path)
end


function _save_current_correlation_plot(path::AbstractString, result)::String
    figure = Figure()
    axis = Axis(
        figure[1, 1],
        xlabel="Time (fs)",
        ylabel="Current correlation (fs⁻²)",
    )
    lines!(axis, result.times, real.(result.correlation); label="Real")
    lines!(axis, result.times, imag.(result.correlation); label="Imaginary")
    axislegend(axis)
    save(String(path), figure)
    return String(path)
end


function _save_current_correlation_rank_plot(path::AbstractString, result)::String
    figure = Figure()
    axis = Axis(
        figure[1, 1],
        xlabel="Time (fs)",
        ylabel="TT rank",
    )
    lines!(axis, result.times, result.maximum_rank; label="Maximum rank")
    lines!(axis, result.times, result.mean_rank; label="Mean rank")
    axislegend(axis)
    save(String(path), figure)
    return String(path)
end
