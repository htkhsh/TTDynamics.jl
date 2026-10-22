using LinearAlgebra
using TTDynamics

isdefined(@__MODULE__, :DEFAULT_CONFIG) || include(joinpath(@__DIR__, "config.jl"))
isdefined(@__MODULE__, :build_problem) || include(joinpath(@__DIR__, "model.jl"))
isdefined(@__MODULE__, :run_current_correlation) || include(joinpath(@__DIR__, "utils.jl"))

_ensure_lattice_frohlich_plotting() =
    isdefined(@__MODULE__, :save_current_plot) ||
    include(joinpath(@__DIR__, "plotting.jl"))

const CORRELATION_OUTPUT_DIRECTORY = joinpath(@__DIR__, "output")
const CORRELATION_CSV_FILENAME = "lattice_frohlich_current_correlation.csv"
const CORRELATION_PLOT_FILENAME = "lattice_frohlich_current_correlation.png"
const CORRELATION_RANK_PLOT_FILENAME = "lattice_frohlich_current_correlation_ranks.png"
const CORRELATION_STATE_FILENAME = "lattice_frohlich_equilibrium.ttbin"
const CORRELATION_METADATA_FILENAME = "lattice_frohlich_equilibrium_metadata.toml"

function current_correlation_main(config=DEFAULT_CONFIG;
                                  correlation_time_fs::Real=CORRELATION_TIME_FS,
                                  output_directory::AbstractString=CORRELATION_OUTPUT_DIRECTORY)
    configure_blas(config)
    println("Lattice Fröhlich HEOM-TT current correlation")
    println("  BLAS threads = $(BLAS.get_num_threads())")
    println("  bath decomposition = $(config.decomposition_method)")
    directory = String(output_directory)
    state_path = joinpath(directory, CORRELATION_STATE_FILENAME)
    metadata_path = joinpath(directory, CORRELATION_METADATA_FILENAME)
    isfile(state_path) || throw(ArgumentError("missing equilibrium state: $state_path"))
    isfile(metadata_path) || throw(ArgumentError("missing equilibrium metadata: $metadata_path"))
    decomposition = decompose_bath(config)
    problem = build_problem(config, decomposition)
    state = load_tt_binary(state_path)
    validate_equilibrium_metadata(read_equilibrium_metadata(metadata_path), config, problem, state)
    println("  step        time_fs       correlation                 max_rank  mean_rank")
    result = run_current_correlation(state, config, problem; duration_fs=correlation_time_fs,
        progress_callback=p -> println("  $(lpad(p.step, 4))  $(lpad(round(p.time_fs; digits=3), 10))  " *
            "$(lpad(string(round(p.correlation; sigdigits=8)), 26))  " *
            "$(lpad(p.maximum_rank, 8))  $(lpad(round(p.mean_rank; digits=2), 9))"))
    mkpath(directory)
    csv_path = joinpath(directory, CORRELATION_CSV_FILENAME)
    plot_path = joinpath(directory, CORRELATION_PLOT_FILENAME)
    rank_plot_path = joinpath(directory, CORRELATION_RANK_PLOT_FILENAME)
    write_current_correlation_csv(csv_path, result; overwrite=true)
    _ensure_lattice_frohlich_plotting()
    current_plotter = Base.invokelatest(getfield, @__MODULE__, :save_current_plot)
    rank_plotter = Base.invokelatest(getfield, @__MODULE__, :save_rank_plot)
    Base.invokelatest(current_plotter, plot_path, result)
    Base.invokelatest(rank_plotter, rank_plot_path, result)
    println("  initial correlation = $(result.correlation[1])")
    println("  final TT ranks = $(tt_ranks(result.state))")
    println("  wrote $csv_path")
    println("  wrote $plot_path")
    println("  wrote $rank_plot_path")
    return (; result, decomposition, problem, state_path, metadata_path, csv_path, plot_path, rank_plot_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    current_correlation_main()
end
