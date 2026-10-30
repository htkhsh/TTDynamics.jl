using LinearAlgebra
using TOML
using TTDynamics
using TTSolver

isdefined(@__MODULE__, :DEFAULT_CONFIG) || include(joinpath(@__DIR__, "config.jl"))
isdefined(@__MODULE__, :build_problem) || include(joinpath(@__DIR__, "model.jl"))
isdefined(@__MODULE__, :run_current_correlation) || include(joinpath(@__DIR__, "utils.jl"))

function current_correlation_main(config=DEFAULT_CONFIG;
                                  correlation_time_fs::Real=CORRELATION_TIME_FS,
                                  output_directory::Union{Nothing,AbstractString}=nothing)
    configure_blas(config)
    println("Peierls HEOM-TT current correlation")
    println("  BLAS threads = $(BLAS.get_num_threads())")
    println("  bath decomposition = $(config.decomposition_method)")
    configured_directory = String(config.output_directory)
    directory = output_directory === nothing ?
        (isabspath(configured_directory) ? configured_directory : joinpath(@__DIR__, configured_directory)) :
        String(output_directory)
    state_path = joinpath(directory, "peierls_equilibrium.ttbin")
    metadata_path = joinpath(directory, "peierls_equilibrium_metadata.toml")
    isfile(state_path) || throw(ArgumentError("missing equilibrium state: $state_path"))
    isfile(metadata_path) || throw(ArgumentError("missing equilibrium metadata: $metadata_path"))

    decomposition = decompose_bath(config)
    problem = build_problem(config, decomposition)
    equilibrium_state = load_tt_binary(state_path)
    validate_equilibrium_metadata(TOML.parsefile(metadata_path), config, problem,
                                  equilibrium_state)
    println("  step        time_fs       correlation                 max_rank  mean_rank")
    result = run_current_correlation(equilibrium_state, config, problem;
        duration_fs=correlation_time_fs,
        progress_callback=progress -> println(
            "  $(lpad(progress.step, 4))  " *
            "$(lpad(round(progress.time_fs; digits=3), 10))  " *
            "$(lpad(string(round(progress.correlation; sigdigits=8)), 26))  " *
            "$(lpad(progress.maximum_rank, 8))  " *
            "$(lpad(round(progress.mean_rank; digits=2), 9))"))

    csv_path = joinpath(directory, "peierls_current_correlation.csv")
    plot_path = joinpath(directory, "peierls_current_correlation.png")
    rank_plot_path = joinpath(directory, "peierls_current_correlation_ranks.png")
    write_current_correlation_csv(csv_path, result)
    isdefined(@__MODULE__, :save_current_plot) ||
        include(joinpath(@__DIR__, "plotting.jl"))
    Base.invokelatest(save_current_plot, plot_path, result)
    Base.invokelatest(save_rank_plot, rank_plot_path, result)
    println("  initial correlation = $(result.correlation[1])")
    println("  final TT ranks = $(tt_ranks(result.state))")
    println("  wrote $csv_path")
    return (; result, decomposition, problem, state_path, metadata_path,
            csv_path, plot_path, rank_plot_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    current_correlation_main()
end
