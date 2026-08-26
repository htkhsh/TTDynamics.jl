function _load_quartic_cairomakie()
    if !isdefined(@__MODULE__, :CairoMakie)
        @eval import CairoMakie
    end
    return Base.eval(@__MODULE__, :CairoMakie)
end

function _quartic_publish_png(
    path::AbstractString,
    render;
    overwrite::Bool=false,
)::String
    return _quartic_publish_png(render, path; overwrite)
end

function _quartic_publish_png(
    render,
    path::AbstractString;
    overwrite::Bool=false,
)::String
    target = String(path)
    !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
    mkpath(dirname(abspath(target)))
    temporary_path = nothing
    try
        base_path, io = mktemp(dirname(abspath(target)))
        close(io)
        rm(base_path)
        temporary_path = "$(base_path).png"
        render(temporary_path)
        isfile(temporary_path) || throw(ArgumentError("plotter did not write a PNG file"))
        !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
        if overwrite
            _quartic_atomic_rename(temporary_path, target)
            temporary_path = nothing
        else
            Base.Filesystem.hardlink(temporary_path, target)
        end
        return target
    finally
        temporary_path === nothing || rm(temporary_path; force=true)
    end
end

function _quartic_line_plot(
    path,
    times,
    series;
    xlabel="Time",
    ylabel,
    title,
    overwrite,
)
    return _quartic_publish_png(path; overwrite) do temporary_path
        cm = _load_quartic_cairomakie()
        figure = Base.invokelatest(cm.Figure; size=(900, 600))
        figure_cell = Base.invokelatest(getindex, figure, 1, 1)
        axis = Base.invokelatest(
            cm.Axis,
            figure_cell;
            xlabel,
            ylabel,
            title,
        )
        for item in series
            Base.invokelatest(
                cm.lines!,
                axis,
                times,
                item.values;
                linewidth=2,
                label=item.label,
            )
        end
        Base.invokelatest(cm.axislegend, axis; position=:rt)
        Base.invokelatest(cm.save, temporary_path, figure)
    end
end

"""
    plot_one_site_result(paths, result[, publication_callback]; overwrite=false)

Lazily load CairoMakie and publish the one-site population, oscillator,
trace, and TT-rank diagnostics without overwriting existing files by default.
`publication_callback(path)` is invoked immediately after each atomic PNG
publication so callers can roll back only outputs they own.
"""
function plot_one_site_result(
    paths,
    result,
    publication_callback=path -> nothing;
    overwrite::Bool=false,
)
    time_count = length(result.times)
    all(
        size(values, 2) == time_count for values in (
            result.populations,
            result.oscillator_q,
            result.oscillator_q2,
            result.oscillator_energy,
        )
    ) || throw(ArgumentError("quartic result columns must match the time grid"))

    _quartic_line_plot(
        paths.populations,
        result.times,
        [(label="electron population", values=result.populations[1, :])];
        ylabel="Population",
        title="One-site electron population",
        overwrite,
    )
    publication_callback(paths.populations)
    _quartic_line_plot(
        paths.oscillator_q,
        result.times,
        [(label="q", values=result.oscillator_q[1, :])];
        ylabel="⟨q⟩",
        title="One-site oscillator position",
        overwrite,
    )
    publication_callback(paths.oscillator_q)
    _quartic_line_plot(
        paths.oscillator_q2,
        result.times,
        [(label="q²", values=result.oscillator_q2[1, :])];
        ylabel="⟨q²⟩",
        title="One-site oscillator squared position",
        overwrite,
    )
    publication_callback(paths.oscillator_q2)
    _quartic_line_plot(
        paths.oscillator_energy,
        result.times,
        [(label="local energy", values=result.oscillator_energy[1, :])];
        ylabel="⟨h_loc⟩",
        title="One-site bare oscillator energy",
        overwrite,
    )
    publication_callback(paths.oscillator_energy)
    _quartic_line_plot(
        paths.trace,
        result.times,
        [
            (label="real", values=real.(result.trace)),
            (label="imaginary", values=imag.(result.trace)),
        ];
        ylabel="Tr(ρ_root)",
        title="Root-ADO trace",
        overwrite,
    )
    publication_callback(paths.trace)
    _quartic_line_plot(
        paths.rank,
        result.times,
        [
            (label="maximum", values=result.maximum_rank),
            (label="mean", values=result.mean_rank),
        ];
        ylabel="TT rank",
        title="HEOM-TT rank diagnostics",
        overwrite,
    )
    publication_callback(paths.rank)
    return paths
end

"""
    plot_two_site_comparison(path, cases; overwrite=false)

Plot both site populations for the harmonic/quartic and bath/no-bath cases.
CairoMakie is loaded only when this function is called.
"""
function plot_two_site_comparison(
    path::AbstractString,
    cases;
    overwrite::Bool=false,
)::String
    isempty(cases) && throw(ArgumentError("at least one comparison case is required"))
    return _quartic_publish_png(path; overwrite) do temporary_path
        cm = _load_quartic_cairomakie()
        figure = Base.invokelatest(cm.Figure; size=(1000, 650))
        figure_cell = Base.invokelatest(getindex, figure, 1, 1)
        axis = Base.invokelatest(
            cm.Axis,
            figure_cell;
            xlabel="Time",
            ylabel="Population",
            title="Two-site quartic HEOM-TT charge transfer",
        )
        for case in cases
            length(case.result.times) == size(case.result.populations, 2) || throw(
                ArgumentError("population columns must match the time grid"),
            )
            for site in axes(case.result.populations, 1)
                Base.invokelatest(
                    cm.lines!,
                    axis,
                    case.result.times,
                    case.result.populations[site, :];
                    linewidth=2,
                    label="$(case.label), site $site",
                )
            end
        end
        Base.invokelatest(cm.axislegend, axis; position=:rt)
        Base.invokelatest(cm.save, temporary_path, figure)
    end
end
