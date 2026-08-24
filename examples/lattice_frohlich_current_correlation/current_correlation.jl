if !isdefined(@__MODULE__, :lattice_frohlich_equilibrate_main)
    include("equilibrate.jl")
end

const DEFAULT_LATTICE_FROHLICH_CORRELATION_TIME_FS = 200.0

function _load_lattice_frohlich_current_plotters()
    if !isdefined(@__MODULE__, :_save_lattice_frohlich_current_plot) ||
       !isdefined(@__MODULE__, :_save_lattice_frohlich_current_rank_plot)
        include(joinpath(@__DIR__, "plotting.jl"))
    end
    return nothing
end

function _default_lattice_frohlich_current_plotter(path, result)
    _load_lattice_frohlich_current_plotters()
    plotter = Base.invokelatest(
        getfield,
        @__MODULE__,
        :_save_lattice_frohlich_current_plot,
    )
    return Base.invokelatest(plotter, path, result)
end

function _default_lattice_frohlich_current_rank_plotter(path, result)
    _load_lattice_frohlich_current_plotters()
    plotter = Base.invokelatest(
        getfield,
        @__MODULE__,
        :_save_lattice_frohlich_current_rank_plot,
    )
    return Base.invokelatest(plotter, path, result)
end

function _lattice_frohlich_current_output_paths(output_directory::AbstractString)
    directory = String(output_directory)
    equilibrium_paths = _lattice_frohlich_equilibration_output_paths(directory)
    return (
        state_path=equilibrium_paths.state_path,
        metadata_path=equilibrium_paths.metadata_path,
        csv_path=joinpath(directory, "lattice_frohlich_current_correlation.csv"),
        png_path=joinpath(directory, "lattice_frohlich_current_correlation.png"),
        rank_png_path=joinpath(directory, "lattice_frohlich_current_correlation_ranks.png"),
    )
end

function _require_lattice_frohlich_equilibrium_inputs(paths)::Nothing
    isfile(paths.state_path) ||
        throw(ArgumentError("equilibrium state file does not exist: $(paths.state_path)"))
    isfile(paths.metadata_path) ||
        throw(ArgumentError("equilibrium metadata file does not exist: $(paths.metadata_path)"))
    return nothing
end

function _require_available_lattice_frohlich_current_outputs(paths;
                                                              overwrite::Bool)::Nothing
    if !overwrite
        for path in (paths.csv_path, paths.png_path, paths.rank_png_path)
            ispath(path) && throw(ArgumentError("target already exists: $path"))
        end
    end
    return nothing
end

"""
    save_lattice_frohlich_current_outputs(paths, result; kwargs) -> NamedTuple

Publish correlation CSV and plots. A non-overwrite publication preflights all
targets and removes every output newly published before a later failure.
"""
function save_lattice_frohlich_current_outputs(
    paths,
    result;
    overwrite::Bool=false,
    correlation_plotter=_default_lattice_frohlich_current_plotter,
    rank_plotter=_default_lattice_frohlich_current_rank_plotter,
)::NamedTuple
    _require_available_lattice_frohlich_current_outputs(paths; overwrite)
    mkpath(dirname(abspath(paths.csv_path)))
    published_paths = String[]
    try
        csv_path = write_lattice_frohlich_current_csv(paths.csv_path, result; overwrite)
        push!(published_paths, csv_path)
        png_path = write_lattice_frohlich_equilibration_png(
            paths.png_path,
            result,
            correlation_plotter;
            overwrite,
        )
        push!(published_paths, png_path)
        rank_png_path = write_lattice_frohlich_equilibration_png(
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
    lattice_frohlich_current_correlation_main(config=DEFAULT_LATTICE_FROHLICH_CURRENT_CONFIG; kwargs)

Reload a validated fixed-time lattice Fröhlich equilibrium checkpoint and
calculate its unsymmetrized particle-current correlation.
"""
function lattice_frohlich_current_correlation_main(
    config=DEFAULT_LATTICE_FROHLICH_CURRENT_CONFIG;
    correlation_time_fs=DEFAULT_LATTICE_FROHLICH_CORRELATION_TIME_FS,
    output_directory=joinpath(@__DIR__, "output"),
    overwrite=true,
    decomposition_builder=decompose_lattice_frohlich_current_bath,
    problem_builder=build_lattice_frohlich_current_model,
    correlation_runner=run_lattice_frohlich_current_correlation,
    plotter=_default_lattice_frohlich_current_plotter,
    rank_plotter=_default_lattice_frohlich_current_rank_plotter,
    progress_io=stdout,
)::NamedTuple
    validate_lattice_frohlich_current_correlation_config(config)
    _lattice_frohlich_correlation_step_count(config, correlation_time_fs)
    paths = _lattice_frohlich_current_output_paths(output_directory)
    _require_lattice_frohlich_equilibrium_inputs(paths)
    _require_available_lattice_frohlich_current_outputs(paths; overwrite)

    decomposition = decomposition_builder(config)
    problem = problem_builder(config, decomposition)
    equilibrium_state = load_tt_binary(paths.state_path)
    metadata = read_lattice_frohlich_equilibrium_metadata(paths.metadata_path)
    validate_lattice_frohlich_equilibrium_state(
        equilibrium_state,
        metadata,
        config,
        decomposition,
        problem,
    )
    println(progress_io, "Loaded equilibrium TT ranks: $(tt_ranks(equilibrium_state))")
    result = correlation_runner(
        equilibrium_state,
        config,
        problem;
        correlation_time_fs,
        progress_callback=p -> _print_lattice_frohlich_current_progress(progress_io, p),
    )
    outputs = save_lattice_frohlich_current_outputs(
        paths,
        result;
        overwrite,
        correlation_plotter=plotter,
        rank_plotter,
    )
    println(progress_io, "Final source TT ranks: $(tt_ranks(result.state))")
    println(progress_io, "Wrote: $(outputs.csv_path)")
    println(progress_io, "Wrote: $(outputs.png_path)")
    println(progress_io, "Wrote: $(outputs.rank_png_path)")
    return (; result, decomposition, problem, state_path=paths.state_path,
            metadata_path=paths.metadata_path, outputs, outputs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    lattice_frohlich_current_correlation_main()
end
