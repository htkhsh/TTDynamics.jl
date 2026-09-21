function _validate_density_matrix(density, dimension, index, tolerance)
    density isa AbstractMatrix || throw(ArgumentError("local density $index must be a matrix"))
    size(density) == (dimension, dimension) || throw(DimensionMismatch(
        "local density $index must have size ($dimension, $dimension)",
    ))
    all(isfinite, density) || throw(ArgumentError("local density $index must be finite"))
    isapprox(density, adjoint(density); atol=tolerance, rtol=0) ||
        throw(ArgumentError("local density $index must be Hermitian"))
    isapprox(real(tr(density)), 1; atol=tolerance, rtol=0) ||
        throw(ArgumentError("local density $index must have unit trace"))
    minimum(eigvals(Hermitian(ComplexF64.(density)))) >= -tolerance ||
        throw(ArgumentError("local density $index must be positive semidefinite"))
end

function _matrix_to_twin_cores(matrix::AbstractMatrix; relative_tolerance::Float64)
    decomposition = svd(ComplexF64.(matrix))
    cutoff = relative_tolerance * maximum(decomposition.S)
    bond_rank = max(count(>(cutoff), decomposition.S), 1)
    ket = reshape(decomposition.U[:, 1:bond_rank], 1, size(matrix, 1), bond_rank)
    bra_matrix = Diagonal(decomposition.S[1:bond_rank]) * decomposition.Vt[1:bond_rank, :]
    bra = reshape(bra_matrix, bond_rank, size(matrix, 2), 1)
    ket, bra
end

function _hierarchy_vacuum_cores(hierarchy_sizes)
    [reshape(ComplexF64[1; zeros(ComplexF64, d - 1)], 1, d, 1)
     for d in hierarchy_sizes]
end

function build_twin_heom_initial_state(
    system::TwinCoreHEOMTTSystem,
    local_density_matrices;
    tol::Float64=1e-12,
    density_tolerance::Float64=1e-10,
)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
    isfinite(density_tolerance) && density_tolerance >= 0 ||
        throw(ArgumentError("density_tolerance must be finite and nonnegative"))
    length(local_density_matrices) == length(system.hilbert_dimensions) ||
        throw(ArgumentError("one local density matrix is required per subsystem"))
    cores = Array{ComplexF64,3}[]
    for (index, (density, dimension)) in enumerate(zip(local_density_matrices, system.hilbert_dimensions))
        _validate_density_matrix(density, dimension, index, density_tolerance)
        ket, bra = _matrix_to_twin_cores(density; relative_tolerance=density_tolerance)
        push!(cores, ket, bra)
    end
    append!(cores, _hierarchy_vacuum_cores(system.hierarchy_sizes))
    tt_round(TTTensor(cores), tol)
end

function _validate_twin_heom_state(state::TTTensor, system::TwinCoreHEOMTTSystem)
    expected = twin_core_heom_dimensions(system)
    tt_dims(state) == expected || throw(DimensionMismatch(
        "twin-core HEOM state dimensions must be $expected; got $(tt_dims(state))",
    ))
end

function root_twin_state(state::TTTensor, system::TwinCoreHEOMTTSystem)::TTTensor
    _validate_twin_heom_state(state, system)
    physical_count = 2 * length(system.hilbert_dimensions)
    environment = ones(ComplexF64, 1)
    for core in Iterators.reverse(state.cores[(physical_count + 1):end])
        environment = ComplexF64.(core[:, 1, :]) * environment
    end
    cores = [ComplexF64.(copy(core)) for core in state.cores[1:physical_count]]
    last_core = cores[end]
    left_rank, dimension, right_rank = size(last_core)
    contracted = reshape(last_core, left_rank * dimension, right_rank) * environment
    cores[end] = reshape(contracted, left_rank, dimension, 1)
    TTTensor(cores)
end

function build_twin_observable(local_operators; tol::Float64=1e-12)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
    isempty(local_operators) && throw(ArgumentError("at least one local observable is required"))
    cores = Array{ComplexF64,3}[]
    for (index, operator) in pairs(local_operators)
        operator isa AbstractMatrix || throw(ArgumentError("local observable $index must be a matrix"))
        size(operator, 1) == size(operator, 2) || throw(DimensionMismatch("local observable $index must be square"))
        all(isfinite, operator) || throw(ArgumentError("local observable $index must be finite"))
        ket, bra = _matrix_to_twin_cores(adjoint(operator); relative_tolerance=tol)
        push!(cores, ket, bra)
    end
    tt_round(TTTensor(cores), tol)
end

function build_twin_trace_observable(system::TwinCoreHEOMTTSystem; tol::Float64=1e-12)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
    identities = [Matrix{ComplexF64}(I, d, d) for d in system.hilbert_dimensions]
    build_twin_observable(identities; tol)
end

function build_twin_local_observable(system::TwinCoreHEOMTTSystem,
                                     subsystem::Integer,
                                     operator::AbstractMatrix;
                                     tol::Float64=1e-12)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
    1 <= subsystem <= length(system.hilbert_dimensions) ||
        throw(ArgumentError("subsystem must index a physical Hilbert core"))
    expected = system.hilbert_dimensions[subsystem]
    size(operator) == (expected, expected) || throw(DimensionMismatch(
        "local observable must have size ($expected, $expected)",
    ))
    factors = [Matrix{ComplexF64}(I, d, d) for d in system.hilbert_dimensions]
    factors[subsystem] = ComplexF64.(operator)
    build_twin_observable(factors; tol)
end

function twin_expectation(observable::TTTensor, state::TTTensor,
                          system::TwinCoreHEOMTTSystem)
    expected = twin_core_physical_dimensions(system)
    tt_dims(observable) == expected || throw(DimensionMismatch(
        "twin observable dimensions must be $expected; got $(tt_dims(observable))",
    ))
    tt_dot(observable, root_twin_state(state, system))
end

function _bounded_hilbert_dimension(dimensions, maximum)
    maximum > 0 || throw(ArgumentError("max_dimension must be positive"))
    total = 1
    for dimension in dimensions
        total <= div(maximum, dimension) || throw(ArgumentError(
            "Hilbert dimension exceeds max_dimension=$maximum",
        ))
        total *= dimension
    end
    total
end

function _kron_linear_index(indices, dimensions)
    index = 1
    for (local_index, dimension) in zip(indices, dimensions)
        index = (index - 1) * dimension + local_index
    end
    index
end

function root_density_matrix(state::TTTensor, system::TwinCoreHEOMTTSystem;
                             max_dimension::Integer=2_000)
    dimensions = system.hilbert_dimensions
    total = _bounded_hilbert_dimension(dimensions, Int(max_dimension))
    root_values = vec(tt_full(root_twin_state(state, system)))
    twin_dimensions = twin_core_physical_dimensions(system)
    density = zeros(ComplexF64, total, total)
    for ket in Iterators.product((Base.OneTo(d) for d in dimensions)...)
        for bra in Iterators.product((Base.OneTo(d) for d in dimensions)...)
            interleaved = Int[]
            for subsystem in eachindex(dimensions)
                push!(interleaved, ket[subsystem], bra[subsystem])
            end
            ket_index = _kron_linear_index(ket, dimensions)
            bra_index = _kron_linear_index(bra, dimensions)
            twin_index = _kron_linear_index(interleaved, twin_dimensions)
            density[ket_index, bra_index] = root_values[twin_index]
        end
    end
    density
end

function build_twin_heom_initial_state(
    system::TwinCoreHEOMTTSystem,
    physical_state::TTTensor;
    tol::Float64=1e-12,
)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
    expected = twin_core_physical_dimensions(system)
    tt_dims(physical_state) == expected || throw(DimensionMismatch(
        "physical twin state dimensions must be $expected; got $(tt_dims(physical_state))",
    ))
    size(first(physical_state.cores), 1) == 1 ||
        throw(ArgumentError("physical twin state left boundary rank must be one"))
    size(last(physical_state.cores), 3) == 1 ||
        throw(ArgumentError("physical twin state right boundary rank must be one"))
    for (index, core) in pairs(physical_state.cores)
        all(isfinite, core) || throw(ArgumentError("physical twin state core $index must be finite"))
    end
    cores = [ComplexF64.(copy(core)) for core in physical_state.cores]
    append!(cores, _hierarchy_vacuum_cores(system.hierarchy_sizes))
    tt_round(TTTensor(cores), tol)
end
