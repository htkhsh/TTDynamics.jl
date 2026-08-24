"""
    measure_holstein_state(rho, trace_observable, population_observables)

Measure site populations, trace, and TT-rank statistics for one HEOM-TT state.
"""
function measure_holstein_state(rho, trace_observable, population_observables)
    populations = real.([tt_dot(observable, rho) for observable in population_observables])
    trace_value = real(tt_dot(trace_observable, rho))
    ranks = tt_ranks(rho)
    return (;
        populations,
        trace=trace_value,
        maximum_rank=maximum(ranks),
        mean_rank=mean(ranks),
    )
end

function _print_holstein_progress(step, time_fs, measurement)
    population_text = join(
        ["P$site=$(round(value, digits=6))" for (site, value) in enumerate(measurement.populations)],
        ", ",
    )
    println(
        "  step $step, t=$(round(time_fs, digits=3)) fs: $population_text, " *
        "trace=$(round(measurement.trace, digits=8)), max rank=$(measurement.maximum_rank)",
    )
end

"""
    run_holstein_dynamics(config, problem)

Propagate the configured initial site state with one Crank-Nicolson tAMEn
solve per saved time step.
"""
function run_holstein_dynamics(config::HolsteinConfig, problem)
    step_count = round(Int, config.final_time_fs / config.time_step_fs)
    save_count = step_count + 1
    times = collect(range(0.0; step=config.time_step_fs, length=save_count))
    populations = zeros(Float64, config.site_count, save_count)
    trace_values = zeros(Float64, save_count)
    maximum_rank = zeros(Int, save_count)
    mean_rank = zeros(Float64, save_count)

    options = Dict(
        :verb => 0,
        :nswp => config.sweep_count,
        :local_iters => config.local_iterations,
        :kickrank => config.kick_rank,
        :time_scheme => "CN",
        :time_error_damp => 100.0,
        :obs => [problem.trace_observable],
    )

    rho = build_initial_state(
        problem.system,
        config.initial_site;
        tol=config.operator_tolerance,
    )
    measurement = measure_holstein_state(
        rho,
        problem.trace_observable,
        problem.population_observables,
    )
    populations[:, 1] = measurement.populations
    trace_values[1] = measurement.trace
    maximum_rank[1] = measurement.maximum_rank
    mean_rank[1] = measurement.mean_rank
    _print_holstein_progress(0, times[1], measurement)

    for time_index in 2:save_count
        space_time_state = tkron(
            rho,
            tt_ones(config.temporal_basis_size; T=ComplexF64),
        )
        space_time_state, time_grid, _, _ = tamen(
            space_time_state,
            config.time_step_fs * problem.liouvillian,
            config.tamen_tolerance,
            options,
        )
        rho = extract_snapshot(space_time_state, time_grid, 1.0, "CN")
        rho = tt_round(rho, config.state_rounding_tolerance)

        measurement = measure_holstein_state(
            rho,
            problem.trace_observable,
            problem.population_observables,
        )
        populations[:, time_index] = measurement.populations
        trace_values[time_index] = measurement.trace
        maximum_rank[time_index] = measurement.maximum_rank
        mean_rank[time_index] = measurement.mean_rank

        step = time_index - 1
        if step % config.progress_interval == 0 || step == step_count
            _print_holstein_progress(step, times[time_index], measurement)
        end
    end

    return (;
        times,
        populations,
        trace=trace_values,
        maximum_rank,
        mean_rank,
    )
end

"""
    write_holstein_population_csv(path, result)

Write all saved populations, trace values, and TT-rank diagnostics as CSV.
"""
function write_holstein_population_csv(path, result)
    site_count = size(result.populations, 1)
    header = join(
        [
            "time_fs",
            ["population_site_$site" for site in 1:site_count]...,
            "trace",
            "max_rank",
            "mean_rank",
        ],
        ",",
    )

    open(path, "w") do io
        println(io, header)
        for time_index in eachindex(result.times)
            row = Any[
                result.times[time_index],
                result.populations[:, time_index]...,
                result.trace[time_index],
                result.maximum_rank[time_index],
                result.mean_rank[time_index],
            ]
            println(io, join(row, ","))
        end
    end
    return path
end
