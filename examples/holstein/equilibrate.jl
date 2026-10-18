using TTDynamics

isdefined(@__MODULE__, :DEFAULT_CONFIG) || include(joinpath(@__DIR__, "config.jl"))
isdefined(@__MODULE__, :build_problem) || include(joinpath(@__DIR__, "model.jl"))
isdefined(@__MODULE__, :run_equilibration) || include(joinpath(@__DIR__, "utils.jl"))
isdefined(@__MODULE__, :save_equilibration_plot) || include(joinpath(@__DIR__, "plotting.jl"))

const EQUILIBRATION_OUTPUT_DIRECTORY = joinpath(@__DIR__, "output")
const EQUILIBRATION_STATE_FILENAME = "holstein_equilibrium.ttbin"
const EQUILIBRATION_METADATA_FILENAME = "holstein_equilibrium_metadata.toml"
const EQUILIBRATION_CSV_FILENAME = "holstein_equilibration.csv"
const EQUILIBRATION_PLOT_FILENAME = "holstein_equilibration_populations.png"

function equilibrate_main(config=DEFAULT_CONFIG;
                          equilibration_time_fs::Real=EQUILIBRATION_TIME_FS,
                          output_directory::AbstractString=EQUILIBRATION_OUTPUT_DIRECTORY)
    println("Holstein HEOM-TT equilibration")
    decomposition = decompose_bath(config)
    problem = build_problem(config, decomposition)
    result = run_equilibration(config, problem; duration_fs=equilibration_time_fs)

    directory = String(output_directory)
    mkpath(directory)
    state_path = joinpath(directory, EQUILIBRATION_STATE_FILENAME)
    metadata_path = joinpath(directory, EQUILIBRATION_METADATA_FILENAME)
    csv_path = joinpath(directory, EQUILIBRATION_CSV_FILENAME)
    plot_path = joinpath(directory, EQUILIBRATION_PLOT_FILENAME)
    save_tt_binary(state_path, result.state; overwrite=true)
    metadata = equilibrium_metadata(config, problem, result.state;
                                    equilibration_time_fs)
    write_equilibrium_metadata(metadata_path, metadata; overwrite=true)
    write_equilibration_csv(csv_path, result; overwrite=true)
    save_equilibration_plot(plot_path, result)

    println("  final trace = $(result.trace[end])")
    println("  final TT ranks = $(tt_ranks(result.state))")
    println("  wrote $csv_path")
    println("  wrote $state_path")
    println("  wrote $metadata_path")
    println("  wrote $plot_path")
    return (; result, decomposition, problem, state_path, metadata_path, csv_path, plot_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    equilibrate_main()
end
