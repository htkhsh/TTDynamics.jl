struct TwinCoreBathCoupling
    subsystem::Int
    operator::Matrix{ComplexF64}
    correlation::ExponentialCorrelation

    function TwinCoreBathCoupling(subsystem::Int,
                                  operator::Matrix{ComplexF64},
                                  correlation::ExponentialCorrelation)
        subsystem > 0 || throw(ArgumentError("subsystem must be positive"))
        _validate_local_operator(operator)
        ishermitian(operator) ||
            throw(ArgumentError("twin-core bath operator must be Hermitian"))
        new(subsystem, operator, correlation)
    end
end

TwinCoreBathCoupling(subsystem::Integer, operator::AbstractMatrix,
                     correlation::ExponentialCorrelation) =
    TwinCoreBathCoupling(Int(subsystem), Matrix{ComplexF64}(operator), correlation)

TwinCoreBathCoupling(coupling::TwinCoreBathCoupling) = coupling

struct TwinCoreHEOMTTSystem
    hamiltonian::TTMatrix{ComplexF64}
    hilbert_dimensions::Vector{Int}
    couplings::Vector{TwinCoreBathCoupling}
    hierarchy_sizes::Vector{Int}
end

function _validate_twin_hamiltonian(hamiltonian::TTMatrix)
    output_dims, input_dims = tt_dims(hamiltonian)
    output_dims == input_dims ||
        throw(DimensionMismatch("Hamiltonian MPO must have square local dimensions"))
    size(first(hamiltonian.cores), 1) == 1 ||
        throw(ArgumentError("Hamiltonian MPO left boundary rank must be one"))
    size(last(hamiltonian.cores), 4) == 1 ||
        throw(ArgumentError("Hamiltonian MPO right boundary rank must be one"))
    for (index, core) in pairs(hamiltonian.cores)
        all(isfinite, core) ||
            throw(ArgumentError("Hamiltonian MPO core $index must be finite"))
    end
    all(>(0), output_dims) ||
        throw(ArgumentError("Hamiltonian local dimensions must be positive"))
    output_dims
end

function TwinCoreHEOMTTSystem(hamiltonian::TTMatrix, couplings, hierarchy_sizes)
    H = _complex_tt_matrix(hamiltonian)
    local_couplings = [TwinCoreBathCoupling(coupling) for coupling in couplings]
    hierarchy = collect(Int, hierarchy_sizes)
    output_dims = _validate_twin_hamiltonian(H)
    expanded_poles = sum(length(c.correlation.rates) for c in local_couplings; init=0)
    length(hierarchy) == expanded_poles || throw(ArgumentError(
        "hierarchy_sizes must provide one positive local size per expanded bath pole",
    ))
    all(>(0), hierarchy) || throw(ArgumentError("hierarchy_sizes must be positive"))
    for coupling in local_couplings
        1 <= coupling.subsystem <= length(output_dims) || throw(ArgumentError(
            "coupling subsystem $(coupling.subsystem) must be in 1:$(length(output_dims))",
        ))
        expected = output_dims[coupling.subsystem]
        size(coupling.operator) == (expected, expected) || throw(DimensionMismatch(
            "coupling subsystem $(coupling.subsystem) expects a $expected x $expected operator",
        ))
    end
    TwinCoreHEOMTTSystem(H, output_dims, local_couplings, hierarchy)
end

function TwinCoreHEOMTTSystem(hamiltonian::TTMatrix, couplings,
                              hierarchy_size::Integer)
    hierarchy_size > 0 || throw(ArgumentError("hierarchy_size must be positive"))
    local_couplings = [TwinCoreBathCoupling(coupling) for coupling in couplings]
    pole_count = sum(length(c.correlation.rates) for c in local_couplings; init=0)
    TwinCoreHEOMTTSystem(hamiltonian, local_couplings,
                         fill(Int(hierarchy_size), pole_count))
end

twin_core_physical_dimensions(system::TwinCoreHEOMTTSystem) =
    reduce(vcat, ([d, d] for d in system.hilbert_dimensions); init=Int[])

twin_core_heom_dimensions(system::TwinCoreHEOMTTSystem) =
    [twin_core_physical_dimensions(system); system.hierarchy_sizes]
