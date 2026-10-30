using LinearAlgebra
using TTDynamics

isdefined(@__MODULE__, :DEFAULT_CONFIG) || include(joinpath(@__DIR__, "config.jl"))
isdefined(@__MODULE__, :build_problem) || include(joinpath(@__DIR__, "model.jl"))
isdefined(@__MODULE__, :run_equilibration) || include(joinpath(@__DIR__, "utils.jl"))

_ensure_lattice_frohlich_plotting() =
    isdefined(@__MODULE__, :save_equilibration_plot) ||
    include(joinpath(@__DIR__, "plotting.jl"))

const EQUILIBRATION_STATE_FILENAME = "lattice_frohlich_equilibrium.ttbin"
const EQUILIBRATION_METADATA_FILENAME = "lattice_frohlich_equilibrium_metadata.toml"
const EQUILIBRATION_CSV_FILENAME = "lattice_frohlich_equilibration.csv"
const EQUILIBRATION_PLOT_FILENAME = "lattice_frohlich_equilibration_populations.png"
const EQUILIBRATION_RANK_PLOT_FILENAME = "lattice_frohlich_equilibration_ranks.png"
const ELECTRON_LATTICE_CSV_FILENAME = "lattice_frohlich_electron_lattice_correlation.csv"
const ELECTRON_LATTICE_PLOT_FILENAME = "lattice_frohlich_electron_lattice_correlation.png"
const SHORT_DISTANCE_ELECTRON_LATTICE_PLOT_FILENAME =
    "lattice_frohlich_electron_lattice_correlation_r0_r1_r2.png"

function equilibrate_main(config=DEFAULT_CONFIG;
                          equilibration_time_fs::Real=EQUILIBRATION_TIME_FS,
                          output_directory::Union{Nothing,AbstractString}=nothing)
    configured_directory = String(config.output_directory)
    directory = output_directory === nothing ?
        (isabspath(configured_directory) ? configured_directory : joinpath(@__DIR__, configured_directory)) :
        String(output_directory)
    configure_blas(config)
    println("Lattice Fröhlich HEOM-TT equilibration")
    print_run_parameters(config, directory;
                         duration_label="equilibration duration",
                         duration_fs=equilibration_time_fs)
    decomposition = decompose_bath(config)
    problem = build_problem(config, decomposition)
    println("  step        time_fs   trace_error    max_rank  mean_rank")
    result = run_equilibration(config, problem; duration_fs=equilibration_time_fs,
        progress_callback=p -> begin
            o = p.observation
            println("  $(lpad(p.step, 4))  $(lpad(round(p.time_fs; digits=3), 10))  " *
                    "$(lpad(string(round(abs(o.trace - 1); sigdigits=8)), 14))  " *
                    "$(lpad(o.maximum_rank, 8))  $(lpad(round(o.mean_rank; digits=2), 9))")
        end)

    mkpath(directory)
    state_path = joinpath(directory, EQUILIBRATION_STATE_FILENAME)
    metadata_path = joinpath(directory, EQUILIBRATION_METADATA_FILENAME)
    csv_path = joinpath(directory, EQUILIBRATION_CSV_FILENAME)
    plot_path = joinpath(directory, EQUILIBRATION_PLOT_FILENAME)
    rank_plot_path = joinpath(directory, EQUILIBRATION_RANK_PLOT_FILENAME)
    electron_lattice_csv_path = joinpath(directory, ELECTRON_LATTICE_CSV_FILENAME)
    electron_lattice_plot_path = joinpath(directory, ELECTRON_LATTICE_PLOT_FILENAME)
    short_distance_electron_lattice_plot_path =
        joinpath(directory, SHORT_DISTANCE_ELECTRON_LATTICE_PLOT_FILENAME)
    save_tt_binary(state_path, result.state; overwrite=true)
    write_equilibrium_metadata(metadata_path, equilibrium_metadata(config, problem, result.state; equilibration_time_fs); overwrite=true)
    write_equilibration_csv(csv_path, result; overwrite=true)
    write_electron_lattice_correlation_csv(electron_lattice_csv_path, result; overwrite=true)
    _ensure_lattice_frohlich_plotting()
    plotter = Base.invokelatest(getfield, @__MODULE__, :save_equilibration_plot)
    Base.invokelatest(plotter, plot_path, result)
    rank_plotter = Base.invokelatest(getfield, @__MODULE__, :save_rank_plot)
    Base.invokelatest(rank_plotter, rank_plot_path, result)
    lattice_plotter = Base.invokelatest(getfield, @__MODULE__, :save_electron_lattice_plot)
    Base.invokelatest(lattice_plotter, electron_lattice_plot_path, result)
    short_distance_plotter = Base.invokelatest(
        getfield, @__MODULE__, :save_short_distance_electron_lattice_plot,
    )
    Base.invokelatest(short_distance_plotter, short_distance_electron_lattice_plot_path, result)
    println("  final trace error = $(abs(result.trace[end] - 1))")
    println("  final TT ranks = $(tt_ranks(result.state))")
    println("  wrote $csv_path")
    println("  wrote $state_path")
    println("  wrote $metadata_path")
    println("  wrote $plot_path")
    println("  wrote $rank_plot_path")
    println("  wrote $electron_lattice_csv_path")
    println("  wrote $electron_lattice_plot_path")
    println("  wrote $short_distance_electron_lattice_plot_path")
    return (; result, decomposition, problem, state_path, metadata_path, csv_path,
            plot_path, rank_plot_path, electron_lattice_csv_path, electron_lattice_plot_path,
            short_distance_electron_lattice_plot_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    equilibrate_main()
end
