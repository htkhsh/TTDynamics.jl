using CairoMakie

"""
    _save_lattice_frohlich_equilibration_plot(path, result) -> String

Render the root-space population of every lattice site against time.
"""
function _save_lattice_frohlich_equilibration_plot(path::AbstractString, result)::String
    length(result.times) == size(result.populations, 2) ||
        throw(ArgumentError("population column count must equal times length"))
    figure = Figure(size=(900, 600))
    axis = Axis(
        figure[1, 1]; xlabel="Time (fs)", ylabel="Population",
        title="Lattice Fröhlich HEOM-TT equilibration populations",
    )
    for site in axes(result.populations, 1)
        lines!(axis, result.times, result.populations[site, :];
               linewidth=2, label="Site $site")
    end
    axislegend(axis; position=:rt)
    save(String(path), figure)
    return String(path)
end
