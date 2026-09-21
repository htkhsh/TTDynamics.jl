"""Local Redfield generators on interleaved MultiCore ket/bra TT spaces."""

struct MultiCoreRedfieldTTSystem
    hamiltonian::TTMatrix{ComplexF64}
    local_hamiltonians::Vector{Matrix{ComplexF64}}
    hilbert_dimensions::Vector{Int}
    couplings::Vector{MultiCoreBathCoupling}
end

function MultiCoreRedfieldTTSystem(hamiltonian::TTMatrix,
                                   local_hamiltonians,
                                   couplings)
    H = _complex_tt_matrix(hamiltonian)
    dimensions = _validate_twin_hamiltonian(H)
    locals = [Matrix{ComplexF64}(local_h) for local_h in local_hamiltonians]
    length(locals) == length(dimensions) || throw(DimensionMismatch(
        "one local Hamiltonian is required per physical subsystem"))
    for (index, local_h) in pairs(locals)
        size(local_h) == (dimensions[index], dimensions[index]) || throw(DimensionMismatch(
            "local Hamiltonian $index has size $(size(local_h)); expected $(dimensions[index]) x $(dimensions[index])"))
        all(isfinite, local_h) || throw(ArgumentError("local Hamiltonian $index must be finite"))
        ishermitian(local_h) || throw(ArgumentError("local Hamiltonian $index must be Hermitian"))
    end
    local_couplings = [MultiCoreBathCoupling(coupling) for coupling in couplings]
    for coupling in local_couplings
        1 <= coupling.subsystem <= length(dimensions) || throw(ArgumentError(
            "coupling subsystem $(coupling.subsystem) must be in 1:$(length(dimensions))"))
        expected = dimensions[coupling.subsystem]
        size(coupling.operator) == (expected, expected) || throw(DimensionMismatch(
            "coupling subsystem $(coupling.subsystem) expects a $expected x $expected operator"))
    end
    MultiCoreRedfieldTTSystem(H, locals, dimensions, local_couplings)
end

function _multicore_redfield_local_matrix(system::MultiCoreRedfieldTTSystem,
                                          replacements...)
    dimensions = reduce(vcat, ([d, d] for d in system.hilbert_dimensions); init=Int[])
    factors = [Matrix{ComplexF64}(I, dimension, dimension) for dimension in dimensions]
    for (index, factor) in replacements
        checkbounds(Bool, factors, index) || throw(ArgumentError(
            "multicore local replacement $index is out of range"))
        size(factor) == size(factors[index]) || throw(DimensionMismatch(
            "multicore replacement $index has size $(size(factor)); expected $(size(factors[index]))"))
        factors[index] = Matrix{ComplexF64}(factor)
    end
    tt_mkron(factors)
end

function _multicore_redfield_bath(correlation::ExponentialCorrelation,
                                  operator::AbstractMatrix)
    source = BathExp(correlation.rates, correlation.coeff_forward, operator)
    RedfieldBath(source)
end

"""
    build_multicore_redfield_dissipator(system; tol=1e-12)

Construct the sum of independent local Redfield dissipators.  HEOMKit builds
each local Redfield jump; the resulting four superoperator terms are embedded
in the interleaved `(ket₁, bra₁, ket₂, bra₂, ...)` TT space.
"""
function build_multicore_redfield_dissipator(system::MultiCoreRedfieldTTSystem;
                                             tol::Float64=1e-12)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
    terms = TTMatrix{ComplexF64}[]
    for coupling in system.couplings
        subsystem = coupling.subsystem
        operator = coupling.operator
        bath = _multicore_redfield_bath(coupling.correlation, operator)
        jump = global_redfield_jump(system.local_hamiltonians[subsystem], operator, bath)
        ket = 2 * subsystem - 1
        bra = 2 * subsystem
        push!(terms, (-1) * _multicore_redfield_local_matrix(system, ket => operator * jump))
        push!(terms, _multicore_redfield_local_matrix(
            system, ket => jump, bra => transpose(operator)))
        push!(terms, (-1) * _multicore_redfield_local_matrix(
            system, bra => transpose(jump' * operator)))
        push!(terms, _multicore_redfield_local_matrix(
            system, ket => operator, bra => transpose(jump')))
    end
    isempty(terms) && return 0 * _multicore_redfield_local_matrix(system)
    result = reduce(+, terms)
    tt_round(result, tol)
end

"""Construct the coherent MultiCore Liouvillian plus its Redfield dissipator."""
function build_multicore_redfield_generator(system::MultiCoreRedfieldTTSystem;
                                            tol::Float64=1e-12)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
    tt_round(build_multicore_liouvillian(system.hamiltonian; tol=tol) +
             build_multicore_redfield_dissipator(system; tol=tol), tol)
end

"""
    propagate_multicore_redfield_tamen(system, state, times; kwargs...)

Propagate a MultiCore Redfield state with TTSolver's tAMEn integrator.  Each
interval is represented by a short temporal TT dimension and interpolated back
to its endpoint, returning one TT state per requested time.
"""
function propagate_multicore_redfield_tamen(system::MultiCoreRedfieldTTSystem,
                                            state::TTTensor,
                                            times;
                                            temporal_basis_size::Integer=3,
                                            tolerance::Real=1e-8,
                                            state_tolerance::Real=1e-12,
                                            generator_tolerance::Real=1e-12,
                                            options=Dict{Symbol,Any}())
    temporal_basis_size >= 2 || throw(ArgumentError("temporal_basis_size must be at least two"))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be finite and positive"))
    isfinite(state_tolerance) && state_tolerance >= 0 || throw(ArgumentError("state_tolerance must be finite and nonnegative"))
    time_grid = collect(Float64, times)
    !isempty(time_grid) || throw(ArgumentError("times must not be empty"))
    all(isfinite, time_grid) && all(diff(time_grid) .>= 0) ||
        throw(ArgumentError("times must be finite and nondecreasing"))
    tt_dims(state) == multicore_physical_dimensions(
        MultiCoreHEOMTTSystem(system.hamiltonian, MultiCoreBathCoupling[], Int[])) ||
        throw(DimensionMismatch("initial state dimensions do not match the MultiCore physical space"))

    generator = build_multicore_redfield_generator(system; tol=Float64(generator_tolerance))
    trajectory = TTTensor[state]
    current = state
    for interval in 2:length(time_grid)
        Δt = time_grid[interval] - time_grid[interval - 1]
        Δt == 0 && (push!(trajectory, current); continue)
        space_time = tkron(current, tt_ones(Int(temporal_basis_size); T=ComplexF64))
        local_options = Dict{Symbol,Any}(options)
        space_time, local_time_grid, _, _ = tamen(
            space_time, Δt * generator, tolerance, local_options,
        )
        current = tt_round(extract_snapshot(space_time, local_time_grid, 1.0, "CN"),
                           Float64(state_tolerance))
        push!(trajectory, current)
    end
    trajectory
end
