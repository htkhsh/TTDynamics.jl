if !isdefined(@__MODULE__, :equilibrate_main)
    include("equilibrate.jl")
end

function _load_current_correlation_plotter()
    if !isdefined(@__MODULE__, :_save_current_correlation_plot) ||
       !isdefined(@__MODULE__, :_save_current_correlation_rank_plot)
        include(joinpath(@__DIR__, "plotting.jl"))
    end
    return nothing
end


function _default_current_correlation_rank_plotter(path, result)
    _load_current_correlation_plotter()
    plotter = Base.invokelatest(
        getfield,
        @__MODULE__,
        :_save_current_correlation_rank_plot,
    )
    return Base.invokelatest(plotter, path, result)
end

function _default_current_correlation_plotter(path, result)
    _load_current_correlation_plotter()
    plotter = Base.invokelatest(getfield, @__MODULE__, :_save_current_correlation_plot)
    return Base.invokelatest(plotter, path, result)
end

const DEFAULT_CORRELATION_TIME_FS = 200.0
const DEFAULT_CURRENT_CORRELATION_OUTPUT_DIRECTORY = joinpath(@__DIR__, "output")
const DEFAULT_CURRENT_CORRELATION_CSV_PATH = joinpath(
    DEFAULT_CURRENT_CORRELATION_OUTPUT_DIRECTORY,
    "holstein_current_correlation.csv",
)
const DEFAULT_CURRENT_CORRELATION_PNG_PATH = joinpath(
    DEFAULT_CURRENT_CORRELATION_OUTPUT_DIRECTORY,
    "holstein_current_correlation.png",
)
const DEFAULT_CURRENT_CORRELATION_RANK_PNG_PATH = joinpath(
    DEFAULT_CURRENT_CORRELATION_OUTPUT_DIRECTORY,
    "holstein_current_correlation_ranks.png",
)

function _current_correlation_output_paths(output_directory::AbstractString)
    directory = String(output_directory)
    equilibrium_paths = _equilibration_output_paths(directory)
    return (
        state_path=equilibrium_paths.state_path,
        metadata_path=equilibrium_paths.metadata_path,
        csv_path=joinpath(directory, basename(DEFAULT_CURRENT_CORRELATION_CSV_PATH)),
        png_path=joinpath(directory, basename(DEFAULT_CURRENT_CORRELATION_PNG_PATH)),
        rank_png_path=joinpath(
            directory,
            basename(DEFAULT_CURRENT_CORRELATION_RANK_PNG_PATH),
        ),
    )
end

function _require_equilibrium_inputs(paths)
    isfile(paths.state_path) ||
        throw(ArgumentError("equilibrium state file does not exist: $(paths.state_path)"))
    isfile(paths.metadata_path) ||
        throw(ArgumentError("equilibrium metadata file does not exist: $(paths.metadata_path)"))
    return nothing
end

function _require_available_current_outputs(paths; overwrite::Bool)::Nothing
    if !overwrite
        for path in (paths.csv_path, paths.png_path, paths.rank_png_path)
            ispath(path) && throw(ArgumentError("target already exists: $path"))
        end
    end
    return nothing
end

function save_current_correlation_outputs(
    paths,
    result;
    overwrite::Bool=false,
    correlation_plotter=_default_current_correlation_plotter,
    rank_plotter=_default_current_correlation_rank_plotter,
)::NamedTuple
    _require_available_current_outputs(paths; overwrite)
    mkpath(dirname(abspath(paths.csv_path)))
    published_paths = String[]
    try
        csv_path = write_current_correlation_csv(paths.csv_path, result; overwrite)
        push!(published_paths, csv_path)
        png_path = write_plot_png(
            paths.png_path,
            result,
            correlation_plotter;
            overwrite,
        )
        push!(published_paths, png_path)
        rank_png_path = write_plot_png(
            paths.rank_png_path,
            result,
            rank_plotter;
            overwrite,
        )
        push!(published_paths, rank_png_path)
        return (; csv_path, png_path, rank_png_path)
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
    current_correlation_main(config=DEFAULT_CONFIG;
                             correlation_time_fs=DEFAULT_CORRELATION_TIME_FS,
                             output_directory=DEFAULT_CURRENT_CORRELATION_OUTPUT_DIRECTORY,
                             overwrite=true) -> NamedTuple

Reload a validated fixed-time equilibrium state and calculate the
unsymmetrized current correlation for 200 fs by default.
"""
function current_correlation_main(config=DEFAULT_CONFIG;
                                  correlation_time_fs::Real=DEFAULT_CORRELATION_TIME_FS,
                                  output_directory::AbstractString=DEFAULT_CURRENT_CORRELATION_OUTPUT_DIRECTORY,
                                  overwrite::Bool=true,
                                  decomposition_builder=_default_decomposition_builder,
                                  problem_builder=_default_problem_builder,
                                  correlation_runner=run_current_correlation,
                                  plotter=_default_current_correlation_plotter,
                                  rank_plotter=_default_current_correlation_rank_plotter,
                                  progress_io::IO=stdout)::NamedTuple
    validate_config(config)
    _correlation_step_count(config, correlation_time_fs)
    paths = _current_correlation_output_paths(output_directory)
    _require_equilibrium_inputs(paths)
    _require_available_current_outputs(paths; overwrite)

    decomposition = decomposition_builder(config)
    problem = problem_builder(config, decomposition)
    equilibrium_state = load_tt_binary(paths.state_path)
    metadata = read_equilibrium_metadata(paths.metadata_path)
    validate_equilibrium_state(equilibrium_state, metadata, config, decomposition, problem)
    _print_tt_rank_vector(progress_io, "Loaded equilibrium TT ranks", equilibrium_state)
    result = correlation_runner(
        equilibrium_state,
        config,
        problem;
        correlation_time_fs,
        progress_callback=progress ->
            _print_current_correlation_progress(progress_io, progress),
    )

    outputs = save_current_correlation_outputs(
        paths,
        result;
        overwrite,
        correlation_plotter=plotter,
        rank_plotter,
    )
    _print_tt_rank_vector(progress_io, "Final source TT ranks", result.state)
    println("  Wrote: $(outputs.csv_path)")
    println("  Wrote: $(outputs.png_path)")
    println("  Wrote: $(outputs.rank_png_path)")
    return (; result, decomposition, problem, state_path=paths.state_path,
            metadata_path=paths.metadata_path, outputs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    current_correlation_main()
end
