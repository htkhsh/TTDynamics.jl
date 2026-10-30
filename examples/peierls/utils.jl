using LinearAlgebra
using Statistics
using TOML
using QFiND
using TTDynamics
using TTSolver

function configure_blas(config)
    config.blas_threads isa Integer && config.blas_threads >= 1 ||
        throw(ArgumentError("blas_threads must be a positive integer"))
    BLAS.set_num_threads(config.blas_threads)
    return BLAS.get_num_threads()
end

function step_count(duration_fs::Real, time_step_fs::Real; allow_zero::Bool=false)
    isfinite(duration_fs) && duration_fs >= 0 ||
        throw(ArgumentError("duration_fs must be finite and nonnegative"))
    isfinite(time_step_fs) && time_step_fs > 0 ||
        throw(ArgumentError("time_step_fs must be finite and positive"))
    !allow_zero && duration_fs == 0 &&
        throw(ArgumentError("duration_fs must contain at least one step"))
    steps = round(Int, duration_fs / time_step_fs)
    isapprox(steps * time_step_fs, duration_fs; rtol=0,
             atol=eps(Float64) * max(1, abs(duration_fs))) ||
        throw(ArgumentError("duration_fs must be an integer multiple of time_step_fs"))
    return steps
end

function initial_heom_state(config, system::HEOMTTSystem)
    n = config.site_count
    size(system.H_sys) == (n, n) || throw(DimensionMismatch("wrong system size"))
    density = if config.initial_condition === :localized
        1 <= config.initial_site <= n || throw(ArgumentError("initial_site is out of range"))
        result = zeros(ComplexF64, n, n)
        result[config.initial_site, config.initial_site] = 1
        result
    elseif config.initial_condition === :gibbs
        config.temperature_K > 0 || throw(ArgumentError("temperature_K must be positive"))
        eigensystem = eigen(Hermitian(system.H_sys))
        β = QFiND.ħ * 1e15 / (QFiND.kb * config.temperature_K)
        weights = exp.(-β .* (eigensystem.values .- minimum(eigensystem.values)))
        weights ./= sum(weights)
        eigensystem.vectors * Diagonal(weights) * eigensystem.vectors'
    else
        throw(ArgumentError("initial_condition must be :localized or :gibbs"))
    end
    ket = reshape(Matrix{ComplexF64}(I, n, n), 1, n, n)
    bra = reshape(ComplexF64.(density), n, n, 1)
    vacuum = [reshape(ComplexF64[1; zeros(size - 1)], 1, size, 1)
              for size in system.nb]
    return TTTensor([ket, bra, vacuum...])
end

function tamen_step(state::TTTensor, problem, config, workspace;
                    tolerance::Real)
    extended = tt_kron(state, tt_ones(config.temporal_basis_size; T=ComplexF64))
    extended, time_grid, _, _, _ = tamen(
        extended, config.time_step_fs * problem.liouvillian, tolerance;
        verb=0, nswp=config.sweep_count, local_iters=config.local_iterations,
        kickrank=config.kick_rank, time_scheme="CN", time_error_damp=100.0,
        correct_2_norm=false,
        conserved_observables=[problem.trace_observable], workspace=workspace,
    )
    return extract_snapshot(extended, time_grid, 1.0, "CN")
end

function measure_density(state::TTTensor, problem)
    density = root_density_matrix(state, problem.system)
    pairs = [(row, column) for row in 1:(size(density, 1) - 1)
             for column in (row + 1):size(density, 2)]
    ranks = tt_ranks(state)
    return (; populations=real.(diag(density)), coherence_pairs=pairs,
            coherences=ComplexF64[density[row, col] for (row, col) in pairs],
            trace=real(tt_dot(problem.trace_observable, state)),
            maximum_rank=maximum(ranks), mean_rank=mean(ranks))
end

function run_equilibration(config, problem; duration_fs::Real,
                           progress_callback=nothing)
    count = step_count(duration_fs, config.time_step_fs)
    state = initial_heom_state(config, problem.system)
    workspace = TAMEnWorkspace(ComplexF64)
    observations = [measure_density(state, problem)]
    progress_callback === nothing || progress_callback((;
        step=0, time_fs=0.0, observation=observations[1]))
    for step in 1:count
        state = tamen_step(state, problem, config, workspace;
                           tolerance=config.equilibration_tamen_tolerance)
        push!(observations, measure_density(state, problem))
        if progress_callback !== nothing &&
           (step == count || step % config.progress_interval == 0)
            progress_callback((; step, time_fs=step * config.time_step_fs,
                               observation=observations[end]))
        end
    end
    first_observation = first(observations)
    return (; times=collect(0:count) .* config.time_step_fs,
            populations=hcat([item.populations for item in observations]...),
            coherence_pairs=first_observation.coherence_pairs,
            coherences=hcat([item.coherences for item in observations]...),
            trace=[item.trace for item in observations],
            maximum_rank=[item.maximum_rank for item in observations],
            mean_rank=[item.mean_rank for item in observations], state)
end

function run_current_correlation(equilibrium_state::TTTensor, config, problem;
                                 duration_fs::Real,
                                 tamen_tolerance::Real=config.correlation_tamen_tolerance,
                                 progress_callback=nothing)
    count = step_count(duration_fs, config.time_step_fs; allow_zero=true)
    current = peierls_current_action(config, problem.system)
    current_adjoint = TTMatrix([permutedims(conj(core), (1, 3, 2, 4))
                                for core in current.cores])
    observable = tt_round(current_adjoint * problem.trace_observable,
                          config.operator_tolerance)
    exact_source = current * equilibrium_state
    state = tt_round(exact_source, config.state_rounding_tolerance)
    workspace = TAMEnWorkspace(ComplexF64)
    observe(ado) = ComplexF64(tt_dot(observable, ado))
    correlation = ComplexF64[observe(exact_source)]
    maximum_rank = Int[maximum(tt_ranks(state))]
    mean_rank = Float64[mean(tt_ranks(state))]
    progress_callback === nothing || progress_callback((;
        step=0, time_fs=0.0, correlation=correlation[1],
        maximum_rank=maximum_rank[1], mean_rank=mean_rank[1]))
    for step in 1:count
        state = tamen_step(state, problem, config, workspace;
                           tolerance=tamen_tolerance)
        push!(correlation, observe(state))
        ranks = tt_ranks(state)
        push!(maximum_rank, maximum(ranks))
        push!(mean_rank, mean(ranks))
        if progress_callback !== nothing &&
           (step == count || step % config.progress_interval == 0)
            progress_callback((; step, time_fs=step * config.time_step_fs,
                               correlation=correlation[end],
                               maximum_rank=maximum_rank[end],
                               mean_rank=mean_rank[end]))
        end
    end
    return (; times=collect(0:count) .* config.time_step_fs, correlation,
            maximum_rank, mean_rank, state)
end

function equilibrium_metadata(config, problem, state; equilibration_time_fs::Real)
    metadata = Dict(
        "identifier" => "TTDynamics Peierls HEOM-TT equilibrium",
        "version" => 1,
        "site_count" => config.site_count,
        "site_energies_cm" => collect(Float64.(config.site_energies_cm)),
        "hopping_cm" => Float64(config.hopping_cm),
        "bath_model" => String(config.bath_model),
        "reorganization_energy_cm" => Float64(config.reorganization_energy_cm),
        "temperature_K" => Float64(config.temperature_K),
        "decomposition_method" => String(config.decomposition_method),
        "pade_order" => config.pade_order,
        "pade_type" => String(config.pade_type),
        "tpsd_tolerance" => Float64(config.tpsd_tolerance),
        "initial_condition" => String(config.initial_condition),
        "initial_site" => config.initial_site,
        "equilibration_time_fs" => Float64(equilibration_time_fs),
        "hierarchy_sizes" => Int.(problem.system.nb),
        "tt_dimensions" => Int.(tt_dims(state)),
    )
    if config.bath_model === :brownian
        metadata["brownian_frequency_cm"] = Float64(config.brownian_frequency_cm)
        metadata["brownian_damping_cm"] = Float64(config.brownian_damping_cm)
    elseif config.bath_model === :drude
        metadata["drude_cutoff_cm"] = Float64(config.drude_cutoff_cm)
    else
        throw(ArgumentError("bath_model must be :brownian or :drude"))
    end
    return metadata
end

function validate_equilibrium_metadata(metadata, config, problem, state)
    expected = equilibrium_metadata(config, problem, state;
                                    equilibration_time_fs=metadata["equilibration_time_fs"])
    for (key, value) in expected
        haskey(metadata, key) && metadata[key] == value ||
            throw(ArgumentError("equilibrium metadata mismatch: $key"))
    end
    return nothing
end

function write_equilibration_csv(path, result)
    open(path, "w") do io
        headers = ["time_fs", ["population_site_$site" for site in
                               axes(result.populations, 1)]...]
        for (row, col) in result.coherence_pairs
            append!(headers, ["coherence_$(row)_$(col)_real",
                             "coherence_$(row)_$(col)_imag"])
        end
        append!(headers, ["trace_error", "max_rank", "mean_rank"])
        println(io, join(headers, ","))
        for index in eachindex(result.times)
            values = Any[result.times[index], result.populations[:, index]...]
            for coherence in result.coherences[:, index]
                append!(values, (real(coherence), imag(coherence)))
            end
            append!(values, (abs(result.trace[index] - 1),
                             result.maximum_rank[index], result.mean_rank[index]))
            println(io, join(values, ","))
        end
    end
    return path
end

function write_current_correlation_csv(path, result)
    open(path, "w") do io
        println(io, "time_fs,correlation_real_fs^-2,correlation_imag_fs^-2,max_rank,mean_rank")
        for index in eachindex(result.times)
            value = result.correlation[index]
            println(io, join((result.times[index], real(value), imag(value),
                             result.maximum_rank[index], result.mean_rank[index]), ","))
        end
    end
    return path
end
