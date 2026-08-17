using LinearAlgebra
using Statistics
using TTSolver
using TTDynamics
using QFiND
using ExpFit
using KaisouEOM
using KaisouEOM: icm2ifs
using CairoMakie

include("utils.jl")

# IMPORTANT: Before using these numerical defaults for production, converge the
# BCF window/tolerance, hierarchy local size, time step, and TT tolerance.
const DEFAULT_CONFIG = HolsteinConfig()

"""
    fit_brownian_bcf(config)

Sample and exponentially fit the finite-temperature Brownian bath correlation
function described by `config`. Its damping field is passed directly as
QFiND's `Γ_Q`, whose corresponding pole decay is `Γ_Q / 2`.
"""
function fit_brownian_bcf(config::HolsteinConfig)
    spectral_density = BrownianSD(
        config.brownian_frequency_cm,
        config.brownian_damping_cm,
        config.reorganization_energy_cm,
    )
    correlation_function = BosonicBCF(
        spectral_density,
        config.temperature_K;
        ub=config.bcf_upper_bound_cm,
    )

    sample_times = collect(range(
        0.0,
        config.bcf_final_time_fs;
        length=config.bcf_sample_count,
    ))
    sample_step = sample_times[2] - sample_times[1]
    samples = correlation_function.(sample_times)
    exponential_fit = ExpFit.esprit(
        samples,
        sample_step,
        config.bcf_fit_tolerance,
    )

    exponents = ComplexF64.(exponential_fit.expon)
    coefficients = ComplexF64.(exponential_fit.coeff)
    isempty(exponents) && error("Brownian BCF fit returned no exponential terms")
    all(isfinite, exponents) || error("Brownian BCF fit returned nonfinite exponents")
    all(isfinite, coefficients) || error("Brownian BCF fit returned nonfinite coefficients")
    # ExpFit and BathExp use C(t) = sum(c[k] * exp(-exponents[k] * t)),
    # hence a decaying expansion has strictly positive real decay rates.
    all(>(0), real.(exponents)) ||
        error("Brownian BCF fit returned a nondecaying exponential")

    fitted_samples = exponential_fit.(sample_times)
    relative_error = norm(fitted_samples - samples) / norm(samples)
    isfinite(relative_error) || error("Brownian BCF fit error is nonfinite")

    return (;
        exponents,
        coefficients,
        relative_error,
        sample_times,
        samples,
        fitted_samples,
    )
end

"""
    build_holstein_heomtt(config, fit)

Construct the periodic Holstein Hamiltonian and its independent site-local
Brownian baths in the HEOM-TT representation.
"""
function build_holstein_heomtt(config::HolsteinConfig, fit)
    H_cm = periodic_holstein_hamiltonian(
        config.site_energies_cm,
        config.hopping_cm,
    )
    H_fs = H_cm * icm2ifs
    baths = [
        BathExp(fit.exponents, fit.coefficients, projector)
        for projector in site_projectors(config.site_count)
    ]
    noise = NoiseExp(baths)
    system = HEOMTTSystem(H_fs, noise, config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)

    initial_state = build_initial_state(
        system,
        config.initial_site;
        tol=config.operator_tolerance,
    )
    println("  Fitted terms per site: $(length(fit.exponents))")
    println("  Total hierarchy cores: $(length(system.nb))")
    println("  Hierarchy local sizes: $(system.nb)")
    println("  Initial TT ranks: $(tt_ranks(initial_state))")

    return (; system, liouvillian, trace_observable, population_observables)
end

"""
    measure_state(rho, trace_observable, population_observables)

Measure site populations, trace, and TT-rank statistics for one HEOM-TT state.
"""
function measure_state(rho, trace_observable, population_observables)
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

function _print_progress(step, time_fs, measurement)
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
    run_dynamics(config, problem)

Propagate the configured initial site state with one Crank-Nicolson tAMEn
solve per saved time step.
"""
function run_dynamics(config::HolsteinConfig, problem)
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
    measurement = measure_state(
        rho,
        problem.trace_observable,
        problem.population_observables,
    )
    populations[:, 1] = measurement.populations
    trace_values[1] = measurement.trace
    maximum_rank[1] = measurement.maximum_rank
    mean_rank[1] = measurement.mean_rank
    _print_progress(0, times[1], measurement)

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

        measurement = measure_state(
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
            _print_progress(step, times[time_index], measurement)
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
    write_population_csv(path, result)

Write all saved populations, trace values, and TT-rank diagnostics as CSV.
"""
function write_population_csv(path, result)
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

"""
    save_diagnostic_plots(output_directory, config, result)

Save population, trace-conservation, and TT-rank plots in `output_directory`.
"""
function save_diagnostic_plots(output_directory, config::HolsteinConfig, result)
    mkpath(output_directory)
    pole_decay_cm = config.brownian_damping_cm / 2

    population_path = joinpath(output_directory, "holstein_brownian_populations.png")
    population_figure = Figure(size=(900, 600))
    population_axis = Axis(
        population_figure[1, 1];
        xlabel="Time [fs]",
        ylabel="Population",
        title="Periodic Holstein HEOM-TT populations\n" *
              "N=$(config.site_count), J=$(config.hopping_cm) cm⁻¹\n" *
              "Ω=$(config.brownian_frequency_cm) cm⁻¹, " *
              "QFiND Γ_Q=$(config.brownian_damping_cm) cm⁻¹ " *
              "(pole decay Γ_Q/2=$pole_decay_cm cm⁻¹), " *
              "λ=$(config.reorganization_energy_cm) cm⁻¹, T=$(config.temperature_K) K",
    )
    for site in 1:config.site_count
        lines!(
            population_axis,
            result.times,
            result.populations[site, :];
            linewidth=2,
            label="site $site",
        )
    end
    axislegend(population_axis; position=:rt)
    save(population_path, population_figure)

    trace_path = joinpath(output_directory, "holstein_brownian_trace.png")
    trace_figure = Figure(size=(700, 450))
    trace_axis = Axis(
        trace_figure[1, 1];
        xlabel="Time [fs]",
        ylabel="Tr[ρ]",
        title="HEOM-TT trace conservation",
    )
    lines!(trace_axis, result.times, result.trace; linewidth=2, label="trace")
    hlines!(trace_axis, [1.0]; linewidth=1, linestyle=:dash, color=:gray)
    axislegend(trace_axis; position=:rt)
    save(trace_path, trace_figure)

    rank_path = joinpath(output_directory, "holstein_brownian_rank.png")
    rank_figure = Figure(size=(700, 450))
    rank_axis = Axis(
        rank_figure[1, 1];
        xlabel="Time [fs]",
        ylabel="TT rank",
        title="HEOM-TT rank evolution",
    )
    lines!(
        rank_axis,
        result.times,
        result.maximum_rank;
        linewidth=2,
        label="maximum rank",
    )
    lines!(
        rank_axis,
        result.times,
        result.mean_rank;
        linewidth=2,
        label="mean rank",
    )
    axislegend(rank_axis; position=:rt)
    save(rank_path, rank_figure)

    return (population_path, trace_path, rank_path)
end

"""
    main(config=DEFAULT_CONFIG)

Fit the bath, build and propagate the periodic Holstein HEOM-TT model, and
write its CSV and plot diagnostics beside this script.
"""
function main(config=DEFAULT_CONFIG)
    validate_config(config)
    pole_decay_cm = config.brownian_damping_cm / 2
    println("Periodic Holstein model with independent Brownian baths")
    println(
        "  Sites=$(config.site_count), J=$(config.hopping_cm) cm⁻¹, " *
        "Ω=$(config.brownian_frequency_cm) cm⁻¹, " *
        "QFiND Γ_Q=$(config.brownian_damping_cm) cm⁻¹ " *
        "(pole decay Γ_Q/2=$pole_decay_cm cm⁻¹), " *
        "λ=$(config.reorganization_energy_cm) cm⁻¹, T=$(config.temperature_K) K",
    )
    println(
        "  Evolution: 0:$(config.time_step_fs):$(config.final_time_fs) fs; " *
        "hierarchy local size=$(config.hierarchy_local_size), " *
        "tAMEn tolerance=$(config.tamen_tolerance)",
    )
    println(
        "  BCF fit: 0:$(config.bcf_final_time_fs) fs with " *
        "$(config.bcf_sample_count) samples at tolerance $(config.bcf_fit_tolerance)",
    )

    fit = fit_brownian_bcf(config)
    println(
        "  BCF fit: $(length(fit.exponents)) terms, relative error=" *
        "$(round(fit.relative_error, sigdigits=5))",
    )
    problem = build_holstein_heomtt(config, fit)
    result = run_dynamics(config, problem)

    output_directory = @__DIR__
    csv_path = write_population_csv(
        joinpath(output_directory, "holstein_brownian_populations.csv"),
        result,
    )
    plot_paths = save_diagnostic_plots(output_directory, config, result)
    trace_drift = maximum(abs.(result.trace .- 1.0))
    println("  Maximum absolute trace drift: $(round(trace_drift, sigdigits=5))")
    println("  Wrote: $csv_path")
    for path in plot_paths
        println("  Wrote: $path")
    end
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
