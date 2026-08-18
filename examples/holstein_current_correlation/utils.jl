using LinearAlgebra
using Statistics
using TTDynamics
using TTSolver
using KaisouEOM: icm2ifs

"""
    periodic_current_operator(config) -> Matrix{ComplexF64}

Return the Hermitian particle-current operator for the periodic Holstein chain.
"""
function periodic_current_operator(config::HolsteinConfig)::Matrix{ComplexF64}
    N = config.site_count
    N >= 2 || throw(ArgumentError("site_count must be at least two"))
    isfinite(config.hopping_cm) || throw(ArgumentError("hopping_cm must be finite"))
    config.hopping_cm >= 0 || throw(ArgumentError("hopping_cm must be nonnegative"))

    scale = config.hopping_cm * icm2ifs
    current = zeros(ComplexF64, N, N)
    for site in 1:(N - 1)
        current[site + 1, site] += 1im * scale
        current[site, site + 1] -= 1im * scale
    end
    if N > 2
        current[1, N] += 1im * scale
        current[N, 1] -= 1im * scale
    end
    return current
end

"""
    build_current_heom_operators(config, problem)

Embed the periodic current as a ket-side HEOM-TT action and as a root-space
observable in the canonical `[ket, bra, hierarchy...]` layout.
"""
function build_current_heom_operators(config::HolsteinConfig, problem)
    current = periodic_current_operator(config)
    N = config.site_count
    I_sys = Matrix{ComplexF64}(I, N, N)
    hierarchy_identities = [Matrix{ComplexF64}(I, n, n) for n in problem.system.nb]
    local_matrices = [current, I_sys, hierarchy_identities...]
    left_action = TTMatrix([
        reshape(ComplexF64.(matrix), 1, size(matrix, 1), size(matrix, 2), 1)
        for matrix in local_matrices
    ])
    left_action = tt_round(left_action, config.operator_tolerance)

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

    liouvillian_dimensions = tt_dims(problem.liouvillian)
    action_dimensions = tt_dims(left_action)
    action_dimensions[1] == liouvillian_dimensions[1] ||
        throw(ArgumentError("current left-action output dimensions must match the Liouvillian"))
    action_dimensions[2] == liouvillian_dimensions[2] ||
        throw(ArgumentError("current left-action input dimensions must match the Liouvillian"))
    tt_dims(observable) == heom_tt_dimensions(problem.system) ||
        throw(ArgumentError("current observable dimensions must match the HEOM system"))

    return (; current, left_action, observable)
end

"""
    propagate_cn_step(rho, liouvillian, config) -> TTTensor

Advance one Crank-Nicolson tAMEn step with the Holstein solver settings.
"""
function propagate_cn_step(rho::TTTensor, liouvillian::TTMatrix, config::HolsteinConfig;
                           trace_observable=nothing)::TTTensor
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
        rho,
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
    propagate_fixed_steps(rho, problem, config, step_count; observe)

Run a fixed number of Crank-Nicolson steps, returning the final state together
with arbitrary values returned by `observe(step, time, state)`.
"""
function propagate_fixed_steps(rho::TTTensor, problem, config::HolsteinConfig,
                               step_count::Integer; observe)
    step_count >= 0 || throw(ArgumentError("step_count must be nonnegative"))
    observations = [observe(0, 0.0, rho)]
    state = rho
    trace_observable = hasproperty(problem, :trace_observable) ?
                       problem.trace_observable : nothing
    for step in 1:step_count
        state = propagate_cn_step(
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
    normalize_heom_state(rho, trace_observable; atol=1e-14) -> TTTensor

Scale every hierarchy component by the complex root trace.
"""
function normalize_heom_state(rho::TTTensor, trace_observable::TTTensor;
                              atol::Real=1e-14)::TTTensor
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    trace_value = tt_dot(trace_observable, rho)
    isfinite(real(trace_value)) && isfinite(imag(trace_value)) ||
        throw(ArgumentError("HEOM trace must be finite"))
    abs(trace_value) > atol || throw(ArgumentError("HEOM trace magnitude must exceed atol"))

    normalized = inv(trace_value) * rho
    normalized_trace = tt_dot(trace_observable, normalized)
    isapprox(normalized_trace, 1; atol=atol) ||
        throw(ArgumentError("normalized HEOM trace is not approximately one"))
    return normalized
end

"""
    measure_heom_state(rho, trace_observable, population_observables)

Return root-space population, trace, and TT-rank diagnostics without relying
on the executable Holstein example.
"""
function measure_heom_state(rho::TTTensor, trace_observable::TTTensor,
                            population_observables)
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
