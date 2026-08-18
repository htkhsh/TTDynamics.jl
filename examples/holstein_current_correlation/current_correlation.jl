using CairoMakie

if !isdefined(@__MODULE__, :equilibrate_main)
    include("equilibrate.jl")
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

function _current_correlation_output_paths(output_directory::AbstractString)
    directory = String(output_directory)
    equilibrium_paths = _equilibration_output_paths(directory)
    return (
        state_path=equilibrium_paths.state_path,
        metadata_path=equilibrium_paths.metadata_path,
        csv_path=joinpath(directory, basename(DEFAULT_CURRENT_CORRELATION_CSV_PATH)),
        png_path=joinpath(directory, basename(DEFAULT_CURRENT_CORRELATION_PNG_PATH)),
    )
end

function _require_equilibrium_inputs(paths)
    isfile(paths.state_path) ||
        throw(ArgumentError("equilibrium state file does not exist: $(paths.state_path)"))
    isfile(paths.metadata_path) ||
        throw(ArgumentError("equilibrium metadata file does not exist: $(paths.metadata_path)"))
    return nothing
end

function _save_current_correlation_plot(path::AbstractString, result)
    figure = Figure()
    axis = Axis(
        figure[1, 1],
        xlabel="Time (fs)",
        ylabel="Current correlation (fs⁻²)",
    )
    lines!(axis, result.times, real.(result.correlation); label="Real")
    lines!(axis, result.times, imag.(result.correlation); label="Imaginary")
    axislegend(axis)
    save(String(path), figure)
    return String(path)
end

"""
    current_correlation_main(config=DEFAULT_CONFIG;
                             correlation_time_fs=DEFAULT_CORRELATION_TIME_FS,
                             output_directory=DEFAULT_CURRENT_CORRELATION_OUTPUT_DIRECTORY,
                             overwrite=false) -> NamedTuple

Reload a validated fixed-time equilibrium state and calculate the
unsymmetrized current correlation for 200 fs by default.
"""
function current_correlation_main(config=DEFAULT_CONFIG;
                                  correlation_time_fs::Real=DEFAULT_CORRELATION_TIME_FS,
                                  output_directory::AbstractString=DEFAULT_CURRENT_CORRELATION_OUTPUT_DIRECTORY,
                                  overwrite::Bool=false,
                                  decomposition_builder=_default_decomposition_builder,
                                  problem_builder=_default_problem_builder,
                                  correlation_runner=run_current_correlation,
                                  plotter=_save_current_correlation_plot)::NamedTuple
    validate_config(config)
    _correlation_step_count(config, correlation_time_fs)
    paths = _current_correlation_output_paths(output_directory)
    _require_equilibrium_inputs(paths)
    if !overwrite
        for path in (paths.csv_path, paths.png_path)
            ispath(path) && throw(ArgumentError("target already exists: $path"))
        end
    end

    decomposition = decomposition_builder(config)
    problem = problem_builder(config, decomposition)
    equilibrium_state = load_tt_binary(paths.state_path)
    metadata = read_equilibrium_metadata(paths.metadata_path)
    validate_equilibrium_state(equilibrium_state, metadata, config, decomposition, problem)
    result = correlation_runner(
        equilibrium_state,
        config,
        problem;
        correlation_time_fs,
    )

    csv_path = write_current_correlation_csv(paths.csv_path, result; overwrite)
    try
        png_path = write_current_correlation_png(paths.png_path, result, plotter; overwrite)
        println("  Final source TT ranks: $(tt_ranks(result.state))")
        println("  Wrote: $csv_path")
        println("  Wrote: $png_path")
        return (; result, decomposition, problem, state_path=paths.state_path,
                metadata_path=paths.metadata_path, csv_path, png_path)
    catch
        !overwrite && rm(csv_path; force=true)
        rethrow()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    current_correlation_main()
end
