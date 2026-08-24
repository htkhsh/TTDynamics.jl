using TTDynamics

isdefined(@__MODULE__, :LatticeFrohlichCurrentCorrelationConfig) || include("config.jl")
isdefined(@__MODULE__, :build_lattice_frohlich_current_model) || include("model.jl")
isdefined(@__MODULE__, :run_lattice_frohlich_equilibration) || include("utils.jl")

const DEFAULT_LATTICE_FROHLICH_CURRENT_CONFIG = LatticeFrohlichCurrentCorrelationConfig()

_default_lattice_frohlich_decomposition_builder(config) =
    decompose_lattice_frohlich_current_bath(config)
_default_lattice_frohlich_problem_builder(config, decomposition) =
    build_lattice_frohlich_current_model(config, decomposition)

function _load_lattice_frohlich_equilibration_plotter()
    if !isdefined(@__MODULE__, :_save_lattice_frohlich_equilibration_plot)
        include(joinpath(@__DIR__, "plotting.jl"))
    end
    return nothing
end

function _default_lattice_frohlich_equilibration_plotter(path, result)
    _load_lattice_frohlich_equilibration_plotter()
    plotter = Base.invokelatest(getfield, @__MODULE__, :_save_lattice_frohlich_equilibration_plot)
    return Base.invokelatest(plotter, path, result)
end

function _lattice_frohlich_equilibration_output_paths(output_directory::AbstractString)
    directory = String(output_directory)
    return (
        state_path=joinpath(directory, "lattice_frohlich_equilibrium.ttbin"),
        metadata_path=joinpath(directory, "lattice_frohlich_equilibrium_metadata.toml"),
        csv_path=joinpath(directory, "lattice_frohlich_equilibration.csv"),
        png_path=joinpath(directory, "lattice_frohlich_equilibration_populations.png"),
    )
end

function _write_staged_lattice_frohlich_metadata(path::AbstractString, metadata, writer;
                                                 overwrite::Bool=false)::String
    target = String(path)
    !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
    staging_directory = mktempdir(dirname(abspath(target)))
    staging_path = joinpath(staging_directory, basename(target))
    try
        writer(staging_path, metadata; overwrite=false)
        isfile(staging_path) || throw(ArgumentError("metadata writer did not write a file"))
        !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
        if overwrite
            _lattice_frohlich_atomic_rename(staging_path, target)
        else
            Base.Filesystem.hardlink(staging_path, target)
        end
        return target
    finally
        rm(staging_directory; recursive=true, force=true)
    end
end

"""
    save_lattice_frohlich_equilibration_outputs(config, decomposition, problem, result; kwargs)

Publish the normalized checkpoint, its TOML metadata, CSV trace, and PNG plot.
With `overwrite=false`, targets are preflighted and any newly published files
are removed if a later publication fails.
"""
function save_lattice_frohlich_equilibration_outputs(
    config::LatticeFrohlichCurrentCorrelationConfig, decomposition, problem, result;
    output_directory::AbstractString=joinpath(@__DIR__, "output"),
    equilibration_time_fs::Real=1000.0,
    overwrite::Bool=false,
    metadata_writer=write_lattice_frohlich_equilibrium_metadata,
    plotter=_default_lattice_frohlich_equilibration_plotter,
)::NamedTuple
    _lattice_frohlich_equilibration_step_count(config, equilibration_time_fs)
    paths = _lattice_frohlich_equilibration_output_paths(output_directory)
    if !overwrite
        for path in values(paths)
            ispath(path) && throw(ArgumentError("target already exists: $path"))
        end
    end
    mkpath(String(output_directory))
    published = String[]
    try
        csv_path = write_lattice_frohlich_equilibration_csv(paths.csv_path, result; overwrite)
        push!(published, csv_path)
        state_path = save_tt_binary(paths.state_path, result.state; overwrite)
        push!(published, state_path)
        metadata = lattice_frohlich_equilibrium_metadata(
            config, decomposition, problem, result.state; equilibration_time_fs,
        )
        metadata_path = _write_staged_lattice_frohlich_metadata(
            paths.metadata_path, metadata, metadata_writer; overwrite,
        )
        push!(published, metadata_path)
        png_path = write_lattice_frohlich_equilibration_png(
            paths.png_path, result, plotter; overwrite,
        )
        push!(published, png_path)
        return (; state_path, metadata_path, csv_path, png_path)
    catch
        if !overwrite
            for path in reverse(published)
                rm(path; force=true)
            end
        end
        rethrow()
    end
end

function lattice_frohlich_equilibrate_main(
    config=DEFAULT_LATTICE_FROHLICH_CURRENT_CONFIG;
    equilibration_time_fs=1000.0,
    output_directory=joinpath(@__DIR__, "output"),
    overwrite=true,
    decomposition_builder=_default_lattice_frohlich_decomposition_builder,
    problem_builder=_default_lattice_frohlich_problem_builder,
    equilibration_runner=run_lattice_frohlich_equilibration,
    plotter=_default_lattice_frohlich_equilibration_plotter,
    progress_io=stdout,
)
    validate_lattice_frohlich_current_correlation_config(config)
    _lattice_frohlich_equilibration_step_count(config, equilibration_time_fs)
    decomposition = decomposition_builder(config)
    problem = problem_builder(config, decomposition)
    result = equilibration_runner(
        config, problem;
        equilibration_time_fs,
        progress_callback=p -> _print_lattice_frohlich_equilibration_progress(progress_io, p),
    )
    paths = save_lattice_frohlich_equilibration_outputs(
        config, decomposition, problem, result;
        output_directory, equilibration_time_fs, overwrite, plotter,
    )
    trace_drift = maximum(abs.(result.trace .- 1.0))
    println(progress_io, "Maximum absolute trace drift: $(round(trace_drift, sigdigits=5))")
    println(progress_io, "Final TT ranks: $(tt_ranks(result.state))")
    for path in values(paths)
        println(progress_io, "Wrote: $path")
    end
    return (; result, decomposition, problem, paths, paths...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    lattice_frohlich_equilibrate_main()
end
