using CairoMakie

const SITE_COUNTS = (11, 15, 19)

function read_current_correlation(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("empty current-correlation CSV: $path"))
    header = split(first(lines), ',')
    real_column = findfirst(==("correlation_real_fs^-2"), header)
    imag_column = findfirst(==("correlation_imag_fs^-2"), header)
    time_column = findfirst(==("time_fs"), header)
    all(index -> !isnothing(index), (time_column, real_column, imag_column)) ||
        throw(ArgumentError("unexpected current-correlation CSV header: $path"))
    rows = [split(line, ',') for line in lines[2:end] if !isempty(line)]
    return (; times=parse.(Float64, getindex.(rows, time_column)),
            real=parse.(Float64, getindex.(rows, real_column)),
            imag=parse.(Float64, getindex.(rows, imag_column)))
end

data = Dict(
    n => read_current_correlation(joinpath(@__DIR__, "output-n$n",
                                             "holstein_current_correlation.csv"))
    for n in SITE_COUNTS
)

figure = Figure(size=(1100, 800))
real_axis = Axis(figure[1, 1], ylabel="Re C(t) (fs⁻²)",
                 title="Holstein current correlation by system size")
imag_axis = Axis(figure[2, 1], xlabel="Time (fs)", ylabel="Im C(t) (fs⁻²)")
colors = Makie.wong_colors()
for (index, n) in enumerate(SITE_COUNTS)
    series = data[n]
    lines!(real_axis, series.times, series.real; label="N = $n", color=colors[index], linewidth=2)
    lines!(imag_axis, series.times, series.imag; label="N = $n", color=colors[index], linewidth=2)
end
axislegend(real_axis; position=:rt)
axislegend(imag_axis; position=:rb)
linkxaxes!(real_axis, imag_axis)

output_path = joinpath(@__DIR__, "output-n19",
                       "holstein_current_correlation_n11_n15_n19_overlay.png")
mkpath(dirname(output_path))
save(output_path, figure)
println("Wrote $output_path")
