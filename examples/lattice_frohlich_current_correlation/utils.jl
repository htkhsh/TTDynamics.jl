using LinearAlgebra
using Statistics
using TTDynamics
using TTSolver
using HEOMKit: icm2ifs

"""
    lattice_frohlich_particle_current(config) -> Matrix{ComplexF64}

Return the hopping-generated Hermitian particle current for the periodic
lattice Fröhlich chain.
"""
function lattice_frohlich_particle_current(
    config::LatticeFrohlichCurrentCorrelationConfig,
)::Matrix{ComplexF64}
    validate_lattice_frohlich_current_correlation_config(config)
    N = config.site_count
    scale = config.hopping_cm * icm2ifs
    current = zeros(ComplexF64, N, N)
    for site in 1:(N - 1)
        current[site + 1, site] += im * scale
        current[site, site + 1] -= im * scale
    end
    if N > 2
        current[1, N] += im * scale
        current[N, 1] -= im * scale
    end
    return current
end

"""
    build_lattice_frohlich_current_operators(config, problem)

Embed the particle current as a ket-side HEOM-TT action and a root-space
observable in the canonical `[ket, bra, hierarchy...]` layout.
"""
function build_lattice_frohlich_current_operators(
    config::LatticeFrohlichCurrentCorrelationConfig,
    problem,
)
    current = lattice_frohlich_particle_current(config)
    N = config.site_count
    I_sys = Matrix{ComplexF64}(I, N, N)
    hierarchy_identities = [Matrix{ComplexF64}(I, n, n) for n in problem.system.nb]
    local_matrices = [current, I_sys, hierarchy_identities...]
    left_action = tt_round(
        TTMatrix([
            reshape(ComplexF64.(matrix), 1, size(matrix, 1), size(matrix, 2), 1)
            for matrix in local_matrices
        ]),
        config.operator_tolerance,
    )

    ket_core = zeros(ComplexF64, 1, N, N)
    for ket in 1:N
        ket_core[1, ket, ket] = 1
    end
    bra_core = reshape(current, N, N, 1)
    vacuum_cores = [reshape(ComplexF64[1; zeros(ComplexF64, n - 1)], 1, n, 1)
                    for n in problem.system.nb]
    observable = tt_round(
        TTTensor([ket_core, bra_core, vacuum_cores...]),
        config.operator_tolerance,
    )

    tt_dims(left_action) == tt_dims(problem.liouvillian) ||
        throw(ArgumentError("current left action dimensions must match the Liouvillian"))
    tt_dims(observable) == heom_tt_dimensions(problem.system) ||
        throw(ArgumentError("current observable dimensions must match the HEOM system"))
    return (; current, left_action, observable)
end

"""
    propagate_lattice_frohlich_cn_step(state, liouvillian, config;
                                       trace_observable=nothing) -> TTTensor

Advance one Crank-Nicolson tAMEn step using the configured solver controls.
"""
function propagate_lattice_frohlich_cn_step(
    state::TTTensor,
    liouvillian::TTMatrix,
    config::LatticeFrohlichCurrentCorrelationConfig;
    trace_observable=nothing,
)::TTTensor
    options = Dict(
        :verb => 0,
        :nswp => config.sweep_count,
        :local_iters => config.local_iterations,
        :kickrank => config.kick_rank,
        :time_scheme => "CN",
        :time_error_damp => 100.0,
    )
    !isnothing(trace_observable) && (options[:obs] = [trace_observable])
    space_time_state = tkron(
        state,
        tt_ones(config.temporal_basis_size; T=ComplexF64),
    )
    space_time_state, time_grid, _, _ = tamen(
        space_time_state,
        config.time_step_fs * liouvillian,
        config.tamen_tolerance,
        options,
    )
    snapshot = extract_snapshot(space_time_state, time_grid, 1.0, "CN")
    return tt_round(snapshot, config.state_rounding_tolerance)
end

"""
    propagate_lattice_frohlich_fixed_steps(state, problem, config, step_count;
                                            observe)

Run a fixed number of Crank-Nicolson steps, including an observation at zero.
"""
function propagate_lattice_frohlich_fixed_steps(
    state::TTTensor,
    problem,
    config::LatticeFrohlichCurrentCorrelationConfig,
    step_count::Integer;
    observe,
)
    step_count >= 0 || throw(ArgumentError("step_count must be nonnegative"))
    observations = [observe(0, 0.0, state)]
    trace_observable = hasproperty(problem, :trace_observable) ?
                       problem.trace_observable : nothing
    for step in 1:step_count
        state = propagate_lattice_frohlich_cn_step(
            state,
            problem.liouvillian,
            config;
            trace_observable,
        )
        push!(observations, observe(step, step * config.time_step_fs, state))
    end
    return (; state, observations)
end

"""
    normalize_lattice_frohlich_heom_state(state, trace_observable; atol=1e-14)

Scale every hierarchy component by the finite, nonzero complex root trace.
"""
function normalize_lattice_frohlich_heom_state(
    state::TTTensor,
    trace_observable::TTTensor;
    atol::Real=1e-14,
)::TTTensor
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    trace_value = tt_dot(trace_observable, state)
    isfinite(real(trace_value)) && isfinite(imag(trace_value)) ||
        throw(ArgumentError("HEOM trace must be finite"))
    abs(trace_value) > atol ||
        throw(ArgumentError("HEOM trace magnitude must exceed atol"))
    return inv(trace_value) * state
end

"""
    measure_lattice_frohlich_heom_state(state, trace_observable,
                                        population_observables)

Return root-space populations, real trace, and TT-rank diagnostics.
"""
function measure_lattice_frohlich_heom_state(
    state::TTTensor,
    trace_observable::TTTensor,
    population_observables,
)
    ranks = tt_ranks(state)
    return (;
        populations=real.([tt_dot(observable, state) for observable in population_observables]),
        trace=real(tt_dot(trace_observable, state)),
        maximum_rank=maximum(ranks),
        mean_rank=mean(ranks),
    )
end

function _lattice_frohlich_equilibration_step_count(
    config::LatticeFrohlichCurrentCorrelationConfig,
    equilibration_time_fs::Real,
)::Int
    isfinite(equilibration_time_fs) && equilibration_time_fs > 0 ||
        throw(ArgumentError("equilibration_time_fs must be finite and positive"))
    ratio = equilibration_time_fs / config.time_step_fs
    step_count = round(Int, ratio)
    isapprox(ratio, step_count; rtol=0, atol=eps(Float64) * max(1, abs(ratio))) ||
        throw(ArgumentError("equilibration_time_fs must be an integral multiple of time_step_fs"))
    step_count >= 1 ||
        throw(ArgumentError("equilibration_time_fs must include at least one time step"))
    return step_count
end

"""
    run_lattice_frohlich_equilibration(config, problem;
                                       initial_state=nothing,
                                       equilibration_time_fs=1000.0,
                                       progress_callback=nothing) -> NamedTuple

Propagate a localized lattice Fröhlich HEOM state and return its normalized
final state with root-space measurements at every step.
"""
function run_lattice_frohlich_equilibration(
    config::LatticeFrohlichCurrentCorrelationConfig,
    problem;
    initial_state=nothing,
    equilibration_time_fs::Real=1000.0,
    progress_callback=nothing,
)::NamedTuple
    validate_lattice_frohlich_current_correlation_config(config)
    step_count = _lattice_frohlich_equilibration_step_count(config, equilibration_time_fs)
    state = isnothing(initial_state) ? build_initial_state(
        problem.system,
        config.initial_site;
        tol=config.operator_tolerance,
    ) : initial_state
    state isa TTTensor || throw(ArgumentError("initial_state must be a TTTensor"))

    start_time_ns = time_ns()
    observe = function(step, time, rho)
        observation = measure_lattice_frohlich_heom_state(
            rho,
            problem.trace_observable,
            problem.population_observables,
        )
        should_report = step == 0 || step == step_count ||
                        step % config.progress_interval == 0
        if !isnothing(progress_callback) && should_report
            progress_callback((;
                step,
                step_count,
                time_fs=time,
                trace=observation.trace,
                maximum_rank=observation.maximum_rank,
                mean_rank=observation.mean_rank,
                elapsed_seconds=(time_ns() - start_time_ns) / 1.0e9,
            ))
        end
        return observation
    end
    propagated = propagate_lattice_frohlich_fixed_steps(
        state,
        problem,
        config,
        step_count;
        observe,
    )
    observations = propagated.observations
    times = collect(0:step_count) .* config.time_step_fs
    length(times) == length(observations) ||
        error("equilibration observation count is inconsistent")
    return (;
        times,
        populations=hcat([observation.populations for observation in observations]...),
        trace=[observation.trace for observation in observations],
        maximum_rank=[observation.maximum_rank for observation in observations],
        mean_rank=[observation.mean_rank for observation in observations],
        state=normalize_lattice_frohlich_heom_state(
            propagated.state,
            problem.trace_observable,
        ),
    )
end

function _print_lattice_frohlich_equilibration_progress(io::IO, p)::Nothing
    println(
        io,
        "Step $(p.step)/$(p.step_count) | time=$(p.time_fs) fs | " *
        "trace=$(p.trace) | max rank=$(p.maximum_rank) | " *
        "mean rank=$(p.mean_rank) | elapsed=$(round(p.elapsed_seconds; digits=2)) s",
    )
    return nothing
end

function _lattice_frohlich_correlation_step_count(
    config::LatticeFrohlichCurrentCorrelationConfig,
    correlation_time_fs::Real,
)::Int
    isfinite(correlation_time_fs) && correlation_time_fs >= 0 ||
        throw(ArgumentError("correlation_time_fs must be finite and nonnegative"))
    ratio = correlation_time_fs / config.time_step_fs
    step_count = round(Int, ratio)
    isapprox(ratio, step_count; rtol=0, atol=eps(Float64) * max(1, abs(ratio))) ||
        throw(ArgumentError("correlation_time_fs must be an integral multiple of time_step_fs"))
    return step_count
end

function _print_lattice_frohlich_current_progress(io::IO, p)::Nothing
    println(
        io,
        "Step $(p.step)/$(p.step_count) | time=$(p.time_fs) fs | " *
        "max rank=$(p.maximum_rank) | mean rank=$(p.mean_rank) | " *
        "elapsed=$(round(p.elapsed_seconds; digits=2)) s",
    )
    return nothing
end

"""
    run_lattice_frohlich_current_correlation(equilibrium_state, config, problem;
                                             correlation_time_fs=200.0,
                                             progress_callback=nothing) -> NamedTuple

Propagate the unnormalized `J * rho_eq` source and record the unsymmetrized
particle-current correlation at every time step.
"""
function run_lattice_frohlich_current_correlation(
    equilibrium_state::TTTensor,
    config::LatticeFrohlichCurrentCorrelationConfig,
    problem;
    correlation_time_fs::Real=200.0,
    progress_callback=nothing,
)::NamedTuple
    validate_lattice_frohlich_current_correlation_config(config)
    step_count = _lattice_frohlich_correlation_step_count(config, correlation_time_fs)
    operators = build_lattice_frohlich_current_operators(config, problem)
    exact_source = operators.left_action * equilibrium_state
    source = tt_round(exact_source, config.state_rounding_tolerance)
    start_time_ns = time_ns()
    observe = function(step, time, state)
        ranks = tt_ranks(state)
        observation = (
            correlation=ComplexF64(tt_dot(operators.observable, state)),
            maximum_rank=maximum(ranks),
            mean_rank=mean(ranks),
        )
        should_report = step == 0 || step == step_count ||
                        step % config.progress_interval == 0
        if !isnothing(progress_callback) && should_report
            progress_callback((;
                step,
                step_count,
                time_fs=time,
                maximum_rank=observation.maximum_rank,
                mean_rank=observation.mean_rank,
                elapsed_seconds=(time_ns() - start_time_ns) / 1.0e9,
            ))
        end
        return observation
    end
    propagated = propagate_lattice_frohlich_fixed_steps(
        source,
        problem,
        config,
        step_count;
        observe,
    )
    observations = propagated.observations
    times = collect(0:step_count) .* config.time_step_fs
    length(times) == length(observations) ||
        error("correlation observation count is inconsistent")
    correlation = ComplexF64[observation.correlation for observation in observations]
    correlation[1] = tt_dot(operators.observable, exact_source)
    return (;
        times,
        correlation,
        maximum_rank=[observation.maximum_rank for observation in observations],
        mean_rank=[observation.mean_rank for observation in observations],
        state=propagated.state,
    )
end
