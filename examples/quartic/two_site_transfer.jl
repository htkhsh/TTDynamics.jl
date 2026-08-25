isdefined(@__MODULE__, :one_site_main) || include(joinpath(@__DIR__, "one_site_relaxation.jl"))

if !isdefined(@__MODULE__, :DEFAULT_QUARTIC_TWO_SITE_CONFIG)
    const DEFAULT_QUARTIC_TWO_SITE_CONFIG = QuarticConfig()
end

function _quartic_two_site_paths(output_directory::AbstractString)
    directory = String(output_directory)
    labels = (
        "harmonic_no_bath",
        "harmonic_bath",
        "quartic_no_bath",
        "quartic_bath",
    )
    return (;
        case_csv=Dict(label => joinpath(directory, "$(label).csv") for label in labels),
        comparison=joinpath(directory, "two_site_population_comparison.png"),
        metadata=joinpath(directory, "two_site_metadata.txt"),
    )
end

function _quartic_metadata_text(cases, fit::ExponentialCorrelation)
    io = IOBuffer()
    println(io, "format = quartic-two-site-v1")
    for case in cases
        println(io, "\n[$(case.label)]")
        for field in fieldnames(QuarticConfig)
            println(io, "$(field) = $(repr(getfield(case.config, field)))")
        end
    end
    println(io, "\n[bath_fit]")
    println(io, "rates = $(repr(fit.rates))")
    println(io, "coeff_forward = $(repr(fit.coeff_forward))")
    println(io, "coeff_backward = $(repr(fit.coeff_backward))")
    println(io, "scales = $(repr(fit.scales))")
    for field in fieldnames(CorrelationFitMetadata)
        println(io, "$(field) = $(repr(getfield(fit.metadata, field)))")
    end
    return String(take!(io))
end

function _write_quartic_text(
    path::AbstractString,
    contents::AbstractString;
    overwrite::Bool=false,
)::String
    target = String(path)
    !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
    mkpath(dirname(abspath(target)))
    temporary_path = nothing
    try
        temporary_path, io = mktemp(dirname(abspath(target)))
        try
            write(io, contents)
        finally
            close(io)
        end
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

"""
    two_site_main(config=DEFAULT_QUARTIC_TWO_SITE_CONFIG; kwargs...)

Compare harmonic/quartic and no-bath/finite-bath charge transfer. The finite
bath correlation is fitted once and reused by both damped cases.
"""
function two_site_main(
    config::QuarticConfig=DEFAULT_QUARTIC_TWO_SITE_CONFIG;
    output_directory::AbstractString=joinpath(@__DIR__, "output", "two_site"),
    overwrite::Bool=false,
    correlation_builder=build_quartic_correlation_fit,
    mode_builder=build_quartic_mode,
    model_builder=build_quartic_model,
    propagator=propagate_quartic_heom,
    csv_writer=write_quartic_csv,
    metadata_writer=_write_quartic_text,
    plotter=plot_two_site_comparison,
    progress_io::IO=stdout,
)
    validate_quartic_config(config)
    config.site_count == 2 || throw(ArgumentError("two_site_main requires site_count = 2"))
    config.bath_lambda > 0 || throw(ArgumentError(
        "two_site_main requires a positive finite-bath lambda for its comparison",
    ))
    paths = _quartic_two_site_paths(output_directory)
    case_labels = [
        "harmonic_no_bath",
        "harmonic_bath",
        "quartic_no_bath",
        "quartic_bath",
    ]
    expected_paths = [paths.case_csv[label] for label in case_labels]
    push!(expected_paths, paths.metadata)
    plotter === nothing || push!(expected_paths, paths.comparison)
    _quartic_preflight_outputs(expected_paths; overwrite)

    harmonic_bath_config = _quartic_workflow_config_with(config; K4=0.0)
    harmonic_no_bath_config = _quartic_workflow_config_with(
        harmonic_bath_config;
        bath_lambda=0.0,
    )
    quartic_bath_config = config
    quartic_no_bath_config = _quartic_workflow_config_with(config; bath_lambda=0.0)
    harmonic_mode = mode_builder(harmonic_bath_config)
    quartic_mode = mode_builder(quartic_bath_config)
    fit = correlation_builder(config)
    _print_quartic_fit(progress_io, fit)

    specifications = [
        (
            label="harmonic_no_bath",
            config=harmonic_no_bath_config,
            mode=harmonic_mode,
            fit=nothing,
        ),
        (
            label="harmonic_bath",
            config=harmonic_bath_config,
            mode=harmonic_mode,
            fit=fit,
        ),
        (
            label="quartic_no_bath",
            config=quartic_no_bath_config,
            mode=quartic_mode,
            fit=nothing,
        ),
        (
            label="quartic_bath",
            config=quartic_bath_config,
            mode=quartic_mode,
            fit=fit,
        ),
    ]
    cases = map(specifications) do specification
        model = model_builder(
            specification.config,
            specification.mode,
            specification.fit;
            initial_site=1,
        )
        result = propagator(model, specification.config)
        case_paths = (; csv=paths.case_csv[specification.label])
        return (;
            specification...,
            model,
            result,
            paths=case_paths,
        )
    end

    mkpath(String(output_directory))
    try
        for case in cases
            csv_writer(case.paths.csv, case.result; overwrite)
        end
        metadata_writer(
            paths.metadata,
            _quartic_metadata_text(cases, fit);
            overwrite,
        )
        plotter === nothing || plotter(paths.comparison, cases; overwrite)
    catch
        if !overwrite
            for path in expected_paths
                rm(path; force=true)
            end
        end
        rethrow()
    end
    for path in expected_paths
        isfile(path) && println(progress_io, "Wrote: $path")
    end
    return (;
        config,
        fit,
        cases,
        comparison_path=paths.comparison,
        metadata_path=paths.metadata,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    two_site_main()
end
