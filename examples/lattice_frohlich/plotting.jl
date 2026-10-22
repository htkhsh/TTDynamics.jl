using CairoMakie

function save_equilibration_plot(path::AbstractString, result)
    figure = Figure(size=(1000, 850))
    population_axis = Axis(figure[1, 1], xlabel="Time (fs)", ylabel="Population",
                           title="Lattice Fröhlich HEOM-TT equilibration")
    for site in axes(result.populations, 1)
        lines!(population_axis, result.times, result.populations[site, :]; linewidth=2, label="Site $site")
    end
    axislegend(population_axis; position=:rt)
    coherence_axis = Axis(figure[2, 1], xlabel="Time (fs)", ylabel="|rho[i,j]|",
                          title="Site-pair coherences")
    for (index, (row, column)) in pairs(result.coherence_pairs)
        lines!(coherence_axis, result.times, abs.(result.coherences[index, :]); linewidth=2,
               label="|rho[$row,$column]|")
    end
    axislegend(coherence_axis; position=:rt, nbanks=2)
    linkxaxes!(population_axis, coherence_axis)
    save(String(path), figure)
    return String(path)
end

function save_current_plot(path::AbstractString, result)
    figure = Figure()
    axis = Axis(figure[1, 1], xlabel="Time (fs)", ylabel="Current correlation (fs⁻²)")
    lines!(axis, result.times, real.(result.correlation); label="Real")
    lines!(axis, result.times, imag.(result.correlation); label="Imaginary")
    axislegend(axis)
    save(String(path), figure)
    return String(path)
end

function save_rank_plot(path::AbstractString, result)
    figure = Figure()
    axis = Axis(figure[1, 1], xlabel="Time (fs)", ylabel="TT rank")
    lines!(axis, result.times, result.maximum_rank; label="Maximum rank")
    lines!(axis, result.times, result.mean_rank; label="Mean rank")
    axislegend(axis)
    save(String(path), figure)
    return String(path)
end
