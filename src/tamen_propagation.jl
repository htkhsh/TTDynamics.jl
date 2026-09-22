function _validated_tamen_observables(observables, dimensions)
    observables === nothing && return TTTensor[]
    observables isa AbstractArray || throw(ArgumentError("conserved_observables must be an array of TTTensor objects"))
    result = collect(observables)
    for (index, observable) in pairs(result)
        observable isa TTTensor || throw(ArgumentError("conserved_observables[$index] must be a TTTensor"))
        tt_dims(observable) == dimensions || throw(DimensionMismatch("conserved_observables[$index] has dimensions $(tt_dims(observable)); expected $dimensions"))
    end
    result
end

function _propagate_tamen_trajectory(generator, state, times;
                                     expected_dimensions,
                                     temporal_basis_size,
                                     tolerance,
                                     state_tolerance,
                                     options,
                                     conserved_observables,
                                     preserve_trace,
                                     trace_observable)
    temporal_basis_size >= 2 || throw(ArgumentError("temporal_basis_size must be at least two"))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be finite and positive"))
    isfinite(state_tolerance) && state_tolerance >= 0 || throw(ArgumentError("state_tolerance must be finite and nonnegative"))
    tt_dims(state) == expected_dimensions || throw(DimensionMismatch("initial state dimensions do not match the expected TT space"))
    time_grid = collect(Float64, times)
    !isempty(time_grid) || throw(ArgumentError("times must not be empty"))
    all(isfinite, time_grid) && all(diff(time_grid) .>= 0) || throw(ArgumentError("times must be finite and nondecreasing"))
    preserve_trace isa Bool || throw(ArgumentError("preserve_trace must be a Bool"))
    explicit = _validated_tamen_observables(conserved_observables, expected_dimensions)
    effective = copy(explicit)
    if preserve_trace
        trace_observable === nothing && throw(ArgumentError("preserve_trace=true requires a trace observable"))
        tt_dims(trace_observable) == expected_dimensions || throw(DimensionMismatch("trace observable dimensions do not match the propagated state"))
        pushfirst!(effective, trace_observable)
    end
    local_options = Dict{Symbol,Any}(options)
    !isempty(effective) && haskey(local_options, :obs) && !(local_options[:obs] === nothing || (local_options[:obs] isa AbstractArray && isempty(local_options[:obs]))) &&
        throw(ArgumentError("conserved_observables/preserve_trace conflicts with options[:obs]"))
    if preserve_trace && get(local_options, :correct_2_norm, false) === true
        throw(ArgumentError("preserve_trace conflicts with options[:correct_2_norm]=true"))
    end
    if preserve_trace && !haskey(local_options, :correct_2_norm)
        local_options[:correct_2_norm] = false
    end
    trajectory = TTTensor[state]
    current = state
    for interval in 2:length(time_grid)
        Δt = time_grid[interval] - time_grid[interval - 1]
        Δt == 0 && (push!(trajectory, current); continue)
        space_time = tkron(current, tt_ones(Int(temporal_basis_size); T=ComplexF64))
        interval_options = Dict{Symbol,Any}(local_options)
        scheme = String(get(interval_options, :time_scheme, "cheb"))
        kwargs = isempty(effective) ? NamedTuple() : (conserved_observables=effective,)
        space_time, local_time_grid, _, _, _ = tamen(space_time, Δt * generator, tolerance,
            interval_options; kwargs...)
        current = tt_round(extract_snapshot(space_time, local_time_grid, 1.0, scheme), Float64(state_tolerance))
        push!(trajectory, current)
    end
    trajectory
end

function propagate_heom_tamen(system::HEOMTTSystem, state::TTTensor, times;
                              temporal_basis_size::Integer=3, tolerance::Real=1e-8,
                              state_tolerance::Real=1e-12, generator_tolerance::Real=1e-12,
                              options=Dict{Symbol,Any}(), conserved_observables=nothing,
                              preserve_trace=false)
    generator, trace_observable, _ = build_heom_liouvillian(system; tol=Float64(generator_tolerance))
    _propagate_tamen_trajectory(generator, state, times;
        expected_dimensions=heom_tt_dimensions(system), temporal_basis_size, tolerance,
        state_tolerance, options, conserved_observables, preserve_trace,
        trace_observable)
end

function _full_multicore_heom_trace_observable(system::MultiCoreHEOMTTSystem; tol::Float64=1e-12)
    physical = build_multicore_trace_observable(system; tol)
    TTTensor(vcat([copy(core) for core in physical.cores],
                  _hierarchy_vacuum_cores(system.hierarchy_sizes)))
end

function propagate_multicore_heom_tamen(system::MultiCoreHEOMTTSystem, state::TTTensor, times;
                                        temporal_basis_size::Integer=3, tolerance::Real=1e-8,
                                        state_tolerance::Real=1e-12, generator_tolerance::Real=1e-12,
                                        options=Dict{Symbol,Any}(), conserved_observables=nothing,
                                        preserve_trace=false)
    generator = build_multicore_heom_generator(system; tol=Float64(generator_tolerance))
    trace_observable = preserve_trace ? _full_multicore_heom_trace_observable(system; tol=Float64(generator_tolerance)) : nothing
    _propagate_tamen_trajectory(generator, state, times;
        expected_dimensions=multicore_heom_dimensions(system), temporal_basis_size, tolerance,
        state_tolerance, options, conserved_observables, preserve_trace,
        trace_observable)
end

function propagate_multicore_redfield_tamen(system::MultiCoreRedfieldTTSystem,
                                            state::TTTensor, times;
                                            temporal_basis_size::Integer=3, tolerance::Real=1e-8,
                                            state_tolerance::Real=1e-12, generator_tolerance::Real=1e-12,
                                            options=Dict{Symbol,Any}(), conserved_observables=nothing,
                                            preserve_trace=false)
    generator = build_multicore_redfield_generator(system; tol=Float64(generator_tolerance))
    trace_system = MultiCoreHEOMTTSystem(system.hamiltonian, MultiCoreBathCoupling[], Int[])
    trace_observable = preserve_trace ? build_multicore_trace_observable(trace_system; tol=Float64(generator_tolerance)) : nothing
    _propagate_tamen_trajectory(generator, state, times;
        expected_dimensions=multicore_physical_dimensions(trace_system), temporal_basis_size,
        tolerance, state_tolerance, options, conserved_observables, preserve_trace,
        trace_observable)
end
