using TTDynamics

if !isdefined(@__MODULE__, :HolsteinConfig)
    include("../holstein/utils.jl")
end
if !isdefined(@__MODULE__, :run_equilibration)
    include("utils.jl")
end

if !isdefined(@__MODULE__, :DEFAULT_CONFIG)
    const DEFAULT_CONFIG = HolsteinConfig()
end

function _load_holstein_builders()
    if !isdefined(@__MODULE__, :decompose_brownian_bcf) ||
       !isdefined(@__MODULE__, :build_holstein_heomtt)
        include(joinpath(@__DIR__, "..", "holstein", "holstein_brownian_heomtt.jl"))
    end
    return nothing
end

function _default_decomposition_builder(config)
    _load_holstein_builders()
    builder = Base.invokelatest(getfield, @__MODULE__, :decompose_brownian_bcf)
    return Base.invokelatest(builder, config)
end

function _default_problem_builder(config, decomposition)
    _load_holstein_builders()
    builder = Base.invokelatest(getfield, @__MODULE__, :build_holstein_heomtt)
    return Base.invokelatest(builder, config, decomposition)
end

function _load_equilibration_plotter()
    if !isdefined(@__MODULE__, :_save_equilibration_population_plot)
        include(joinpath(@__DIR__, "plotting.jl"))
    end
    return nothing
end

function _default_equilibration_plotter(path, result)
    _load_equilibration_plotter()
    plotter = Base.invokelatest(getfield, @__MODULE__, :_save_equilibration_population_plot)
    return Base.invokelatest(plotter, path, result)
end

const DEFAULT_EQUILIBRATION_TIME_FS = 1000.0
const DEFAULT_EQUILIBRATION_OUTPUT_DIRECTORY = joinpath(@__DIR__, "output")
const DEFAULT_EQUILIBRATION_STATE_PATH = joinpath(
    DEFAULT_EQUILIBRATION_OUTPUT_DIRECTORY,
    "holstein_equilibrium.ttbin",
)
const DEFAULT_EQUILIBRATION_METADATA_PATH = joinpath(
    DEFAULT_EQUILIBRATION_OUTPUT_DIRECTORY,
    "holstein_equilibrium_metadata.toml",
)
const DEFAULT_EQUILIBRATION_CSV_PATH = joinpath(
    DEFAULT_EQUILIBRATION_OUTPUT_DIRECTORY,
    "holstein_equilibration.csv",
)
const DEFAULT_EQUILIBRATION_PNG_PATH = joinpath(
    DEFAULT_EQUILIBRATION_OUTPUT_DIRECTORY,
    "holstein_equilibration_populations.png",
)

function _equilibration_output_paths(output_directory::AbstractString)
    directory = String(output_directory)
    return (
        state_path=joinpath(directory, basename(DEFAULT_EQUILIBRATION_STATE_PATH)),
        metadata_path=joinpath(directory, basename(DEFAULT_EQUILIBRATION_METADATA_PATH)),
        csv_path=joinpath(directory, basename(DEFAULT_EQUILIBRATION_CSV_PATH)),
        png_path=joinpath(directory, basename(DEFAULT_EQUILIBRATION_PNG_PATH)),
    )
end

function _write_staged_equilibration_metadata(path::AbstractString, metadata, metadata_writer;
                                              overwrite::Bool=false)::String
    target = String(path)
    !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
    temporary_directory = mktempdir(dirname(abspath(target)))
    temporary_path = joinpath(temporary_directory, basename(target))
    try
        metadata_writer(temporary_path, metadata; overwrite=false)
        isfile(temporary_path) || throw(ArgumentError("metadata writer did not write a file"))
        !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
        if overwrite
            _equilibrium_atomic_rename(temporary_path, target)
        else
            Base.Filesystem.hardlink(temporary_path, target)
        end
        return target
    finally
        rm(temporary_directory; recursive=true, force=true)
    end
end

"""
    save_equilibration_outputs(config, decomposition, problem, result;
                               output_directory=DEFAULT_EQUILIBRATION_OUTPUT_DIRECTORY,
                               equilibration_time_fs=DEFAULT_EQUILIBRATION_TIME_FS,
                               overwrite=false) -> NamedTuple

Publish equilibration diagnostics, a TT binary, matching metadata, and a
population plot. For a non-overwrite save, a publication failure removes every
newly written output. An overwrite save cannot restore replaced outputs; use a
fresh output directory for production replacement runs.
"""
function save_equilibration_outputs(config::HolsteinConfig, decomposition, problem, result;
                                    output_directory::AbstractString=DEFAULT_EQUILIBRATION_OUTPUT_DIRECTORY,
                                    equilibration_time_fs::Real=DEFAULT_EQUILIBRATION_TIME_FS,
                                    overwrite::Bool=false,
                                    metadata_writer=write_equilibrium_metadata,
                                    plotter=_default_equilibration_plotter)::NamedTuple
    _equilibration_step_count(config, equilibration_time_fs)
    paths = _equilibration_output_paths(output_directory)
    if !overwrite
        for path in values(paths)
            ispath(path) && throw(ArgumentError("target already exists: $path"))
        end
    end

    mkpath(String(output_directory))
    published_paths = String[]
    try
        csv_path = write_equilibration_csv(paths.csv_path, result; overwrite)
        push!(published_paths, csv_path)
        state_path = save_tt_binary(paths.state_path, result.state; overwrite)
        push!(published_paths, state_path)
        metadata = equilibrium_metadata(
            config,
            decomposition,
            problem,
            result.state;
            equilibration_time_fs,
        )
        metadata_path = _write_staged_equilibration_metadata(
            paths.metadata_path,
            metadata,
            metadata_writer;
            overwrite,
        )
        push!(published_paths, metadata_path)
        png_path = write_plot_png(paths.png_path, result, plotter; overwrite)
        push!(published_paths, png_path)
        return (; state_path, metadata_path, csv_path, png_path)
    catch
        if !overwrite
            for path in reverse(published_paths)
                rm(path; force=true)
            end
        end
        rethrow()
    end
end

"""
    equilibrate_main(config=DEFAULT_CONFIG;
                     equilibration_time_fs=DEFAULT_EQUILIBRATION_TIME_FS,
                     output_directory=DEFAULT_EQUILIBRATION_OUTPUT_DIRECTORY,
                     overwrite=false) -> NamedTuple

Build the default Holstein HEOM problem, equilibrate it for 1000 fs by
default, and save a binary state with matching TOML and CSV diagnostics.
"""
function equilibrate_main(config=DEFAULT_CONFIG;
                          equilibration_time_fs::Real=DEFAULT_EQUILIBRATION_TIME_FS,
                          output_directory::AbstractString=DEFAULT_EQUILIBRATION_OUTPUT_DIRECTORY,
                          overwrite::Bool=false,
                          decomposition_builder=_default_decomposition_builder,
                          problem_builder=_default_problem_builder,
                          equilibration_runner=run_equilibration,
                          plotter=_default_equilibration_plotter)::NamedTuple
    validate_config(config)
    _equilibration_step_count(config, equilibration_time_fs)
    decomposition = decomposition_builder(config)
    problem = problem_builder(config, decomposition)
    result = equilibration_runner(
        config,
        problem;
        equilibration_time_fs,
    )
    paths = save_equilibration_outputs(
        config,
        decomposition,
        problem,
        result;
        output_directory,
        equilibration_time_fs,
        overwrite,
        plotter,
    )
    trace_drift = maximum(abs.(result.trace .- 1.0))
    saved_trace = tt_dot(problem.trace_observable, result.state)
    println("  Maximum absolute trace drift: $(round(trace_drift, sigdigits=5))")
    println("  Saved state trace: $(round(saved_trace, sigdigits=8))")
    println("  Final TT ranks: $(tt_ranks(result.state))")
    println("  Wrote: $(paths.csv_path)")
    println("  Wrote: $(paths.state_path)")
    println("  Wrote: $(paths.metadata_path)")
    println("  Wrote: $(paths.png_path)")
    return (; result, decomposition, problem, paths...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    equilibrate_main()
end
