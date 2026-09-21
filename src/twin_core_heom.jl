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

function _twin_bond_identity(rank::Int, dimension::Int)
    core = zeros(ComplexF64, rank, dimension, dimension, rank)
    for bond in 1:rank, local_index in 1:dimension
        core[bond, local_index, local_index, bond] = 1
    end
    core
end

function _twin_left_embedding(hamiltonian::TTMatrix)
    cores = Array{ComplexF64,4}[]
    for core in hamiltonian.cores
        dimension = size(core, 2)
        push!(cores, ComplexF64.(core))
        push!(cores, _twin_bond_identity(size(core, 4), dimension))
    end
    TTMatrix(cores)
end

function _twin_right_embedding(hamiltonian::TTMatrix)
    cores = Array{ComplexF64,4}[]
    for core in hamiltonian.cores
        dimension = size(core, 2)
        push!(cores, _twin_bond_identity(size(core, 1), dimension))
        push!(cores, ComplexF64.(permutedims(core, (1, 3, 2, 4))))
    end
    TTMatrix(cores)
end

function build_twin_liouvillian(hamiltonian::TTMatrix; tol::Float64=1e-12)
    isfinite(tol) && tol >= 0 ||
        throw(ArgumentError("tol must be finite and nonnegative"))
    _validate_twin_hamiltonian(hamiltonian)
    left = _twin_left_embedding(hamiltonian)
    right = _twin_right_embedding(hamiltonian)
    tt_round(-1im * (left - right), tol)
end

function _twin_core_local_matrix(system::TwinCoreHEOMTTSystem, replacements...)
    dimensions = twin_core_heom_dimensions(system)
    factors = [Matrix{ComplexF64}(I, d, d) for d in dimensions]
    for (index, factor) in replacements
        checkbounds(Bool, factors, index) ||
            throw(ArgumentError("twin-core local replacement $index is out of range"))
        size(factor) == size(factors[index]) || throw(DimensionMismatch(
            "twin-core replacement $index has size $(size(factor)); expected $(size(factors[index]))",
        ))
        factors[index] = ComplexF64.(factor)
    end
    tt_mkron(factors)
end

function _extend_twin_liouvillian(system::TwinCoreHEOMTTSystem; tol::Float64)
    physical = build_twin_liouvillian(system.hamiltonian; tol)
    cores = [copy(core) for core in physical.cores]
    for dimension in system.hierarchy_sizes
        identity_matrix = Matrix{ComplexF64}(I, dimension, dimension)
        push!(cores, reshape(identity_matrix, 1, dimension, dimension, 1))
    end
    TTMatrix(cores)
end

function build_twin_heom_generator(system::TwinCoreHEOMTTSystem;
                                   tol::Float64=1e-12)
    isfinite(tol) && tol >= 0 ||
        throw(ArgumentError("tol must be finite and nonnegative"))
    generator = _extend_twin_liouvillian(system; tol)
    hierarchy_index = 0
    physical_core_count = 2 * length(system.hilbert_dimensions)
    for coupling in system.couplings
        ket_core = 2 * coupling.subsystem - 1
        bra_core = 2 * coupling.subsystem
        correlation = coupling.correlation
        for pole in eachindex(correlation.rates)
            hierarchy_index += 1
            hierarchy_core = physical_core_count + hierarchy_index
            dimension = system.hierarchy_sizes[hierarchy_index]
            number = diagm(0 => ComplexF64.(0:(dimension - 1)))
            lower = diagm(1 => sqrt.(ComplexF64.(1:(dimension - 1))))
            raise = diagm(-1 => sqrt.(ComplexF64.(1:(dimension - 1))))
            V = coupling.operator
            decay = _twin_core_local_matrix(system, hierarchy_core => number)
            upward = _twin_core_local_matrix(system, ket_core => V, hierarchy_core => lower) -
                     _twin_core_local_matrix(system, bra_core => transpose(V), hierarchy_core => lower)
            downward = correlation.coeff_forward[pole] *
                       _twin_core_local_matrix(system, ket_core => V, hierarchy_core => raise) -
                       correlation.coeff_backward[pole] *
                       _twin_core_local_matrix(system, bra_core => transpose(V), hierarchy_core => raise)
            generator -= correlation.rates[pole] * decay
            generator -= 1im * correlation.scales[pole] * upward
            generator -= 1im / correlation.scales[pole] * downward
            generator = tt_round(generator, tol)
        end
    end
    tt_round(generator, tol)
end
