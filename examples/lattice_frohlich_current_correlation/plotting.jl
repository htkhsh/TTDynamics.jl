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

"""Render real and imaginary particle-current correlation components."""
function _save_lattice_frohlich_current_plot(path::AbstractString, result)::String
    figure = Figure()
    axis = Axis(
        figure[1, 1];
        xlabel="Time (fs)",
        ylabel="Current correlation (fs⁻²)",
        title="Lattice Fröhlich particle-current correlation",
    )
    lines!(axis, result.times, real.(result.correlation); label="Real")
    lines!(axis, result.times, imag.(result.correlation); label="Imaginary")
    axislegend(axis)
    save(String(path), figure)
    return String(path)
end

"""Render maximum and mean TT ranks during current propagation."""
function _save_lattice_frohlich_current_rank_plot(path::AbstractString, result)::String
    figure = Figure()
    axis = Axis(
        figure[1, 1];
        xlabel="Time (fs)",
        ylabel="TT rank",
        title="Lattice Fröhlich current-correlation TT ranks",
    )
    lines!(axis, result.times, result.maximum_rank; label="Maximum rank")
    lines!(axis, result.times, result.mean_rank; label="Mean rank")
    axislegend(axis)
    save(String(path), figure)
    return String(path)
end
