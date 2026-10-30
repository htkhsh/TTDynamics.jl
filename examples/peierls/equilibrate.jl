using LinearAlgebra
using TOML
using TTDynamics
using TTSolver

isdefined(@__MODULE__, :DEFAULT_CONFIG) || include(joinpath(@__DIR__, "config.jl"))
isdefined(@__MODULE__, :build_problem) || include(joinpath(@__DIR__, "model.jl"))
isdefined(@__MODULE__, :run_equilibration) || include(joinpath(@__DIR__, "utils.jl"))

function equilibrate_main(config=DEFAULT_CONFIG;
                          equilibration_time_fs::Real=EQUILIBRATION_TIME_FS,
                          output_directory::Union{Nothing,AbstractString}=nothing)
    configure_blas(config)
    println("Peierls HEOM-TT equilibration")
    println("  BLAS threads = $(BLAS.get_num_threads())")
    println("  bath decomposition = $(config.decomposition_method)")
    decomposition = decompose_bath(config)
    problem = build_problem(config, decomposition)
    println("  step        time_fs   trace_error    max_rank  mean_rank")
    result = run_equilibration(config, problem;
        duration_fs=equilibration_time_fs,
        progress_callback=progress -> begin
            observation = progress.observation
            println("  $(lpad(progress.step, 4))  " *
                    "$(lpad(round(progress.time_fs; digits=3), 10))  " *
                    "$(lpad(string(round(abs(observation.trace - 1); sigdigits=8)), 14))  " *
                    "$(lpad(observation.maximum_rank, 8))  " *
                    "$(lpad(round(observation.mean_rank; digits=2), 9))")
        end)

    configured_directory = String(config.output_directory)
    directory = output_directory === nothing ?
        (isabspath(configured_directory) ? configured_directory : joinpath(@__DIR__, configured_directory)) :
        String(output_directory)
    mkpath(directory)
    state_path = joinpath(directory, "peierls_equilibrium.ttbin")
    metadata_path = joinpath(directory, "peierls_equilibrium_metadata.toml")
    csv_path = joinpath(directory, "peierls_equilibration.csv")
    plot_path = joinpath(directory, "peierls_equilibration.png")
    rank_plot_path = joinpath(directory, "peierls_equilibration_ranks.png")
    save_tt_binary(state_path, result.state; overwrite=true)
    open(metadata_path, "w") do io
        TOML.print(io, equilibrium_metadata(config, problem, result.state;
                                            equilibration_time_fs))
    end
    write_equilibration_csv(csv_path, result)
    isdefined(@__MODULE__, :save_equilibration_plot) ||
        include(joinpath(@__DIR__, "plotting.jl"))
    Base.invokelatest(save_equilibration_plot, plot_path, result)
    Base.invokelatest(save_rank_plot, rank_plot_path, result)
    println("  final trace error = $(abs(result.trace[end] - 1))")
    println("  final TT ranks = $(tt_ranks(result.state))")
    println("  wrote $state_path")
    return (; result, decomposition, problem, state_path, metadata_path,
            csv_path, plot_path, rank_plot_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    equilibrate_main()
end
