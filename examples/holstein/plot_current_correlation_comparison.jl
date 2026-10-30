using CairoMakie

function read_model_current_correlation(path::AbstractString)
    rows = [split(line, ',') for line in readlines(path)[2:end] if !isempty(line)]
    return (; times=parse.(Float64, getindex.(rows, 1)),
            real=parse.(Float64, getindex.(rows, 2)),
            imag=parse.(Float64, getindex.(rows, 3)))
end

holstein = read_model_current_correlation(joinpath(
    @__DIR__, "output-n15-re800", "holstein_current_correlation.csv",
))
lattice_frohlich = read_model_current_correlation(joinpath(
    @__DIR__, "..", "lattice_frohlich", "output-n15-re800",
    "lattice_frohlich_current_correlation.csv",
))

figure = Figure(size=(1100, 800))
real_axis = Axis(figure[1, 1], ylabel="Re C(t) (fs⁻²)",
                 title="N=15 current correlation, reorganization energy = 800 cm⁻¹")
imag_axis = Axis(figure[2, 1], xlabel="Time (fs)", ylabel="Im C(t) (fs⁻²)")

for (axis, component) in ((real_axis, :real), (imag_axis, :imag))
    lines!(axis, holstein.times, getproperty(holstein, component);
           label="Holstein (0–600 fs)", color=:dodgerblue, linewidth=2)
    lines!(axis, lattice_frohlich.times, getproperty(lattice_frohlich, component);
           label="Lattice Fröhlich (0–800 fs)", color=:darkorange, linewidth=2)
    axislegend(axis; position=:rt)
end
linkxaxes!(real_axis, imag_axis)

output_directory = joinpath(@__DIR__, "output-n15-re800")
correlation_output_path = joinpath(
    output_directory, "holstein_lattice_frohlich_current_correlation_overlay.png",
)
save(correlation_output_path, figure)
println("Wrote $correlation_output_path")

function cumulative_diffusivity(times, correlation_real)
    diffusivity = zeros(length(times))
    for index in 2:length(times)
        dt = times[index] - times[index - 1]
        diffusivity[index] = diffusivity[index - 1] +
                             0.5 * dt * (correlation_real[index - 1] + correlation_real[index])
    end
    return diffusivity
end

max_diffusivity_time_fs = 600.0
holstein_end = findlast(holstein.times .<= max_diffusivity_time_fs)
lattice_frohlich_end = findlast(lattice_frohlich.times .<= max_diffusivity_time_fs)
holstein_diffusivity = cumulative_diffusivity(
    holstein.times[1:holstein_end], holstein.real[1:holstein_end],
)
lattice_frohlich_diffusivity = cumulative_diffusivity(
    lattice_frohlich.times[1:lattice_frohlich_end],
    lattice_frohlich.real[1:lattice_frohlich_end],
)

diffusivity_figure = Figure(size=(1100, 600))
diffusivity_axis = Axis(
    diffusivity_figure[1, 1], xlabel="Time (fs)",
    ylabel="Diffusivity (a²/fs)",
    title="N=15 diffusivity, reorganization energy = 800 cm⁻¹",
    limits=(0, max_diffusivity_time_fs, nothing, nothing),
)
lines!(diffusivity_axis, holstein.times[1:holstein_end], holstein_diffusivity;
       label="Holstein", color=:dodgerblue, linewidth=2)
lines!(diffusivity_axis, lattice_frohlich.times[1:lattice_frohlich_end],
       lattice_frohlich_diffusivity;
       label="Lattice Fröhlich", color=:darkorange, linewidth=2)
axislegend(diffusivity_axis; position=:rt)

diffusivity_output_path = joinpath(
    output_directory, "holstein_lattice_frohlich_diffusivity_600fs.png",
)
save(diffusivity_output_path, diffusivity_figure)
println("Wrote $diffusivity_output_path")
