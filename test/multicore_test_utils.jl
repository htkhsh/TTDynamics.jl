function kron_linear_index(indices, dimensions)
    index = 1
    for (local_index, dimension) in zip(indices, dimensions)
        index = (index - 1) * dimension + local_index
    end
    index
end

function twin_permutation(dimensions)
    total = prod(dimensions)
    twin_dimensions = reduce(vcat, ([d, d] for d in dimensions); init=Int[])
    permutation = zeros(ComplexF64, total^2, total^2)
    ranges = (Base.OneTo(d) for d in dimensions)
    for ket in Iterators.product(ranges...)
        for bra in Iterators.product((Base.OneTo(d) for d in dimensions)...)
            ket_index = kron_linear_index(ket, dimensions)
            bra_index = kron_linear_index(bra, dimensions)
            interleaved = Int[]
            for subsystem in eachindex(dimensions)
                push!(interleaved, ket[subsystem], bra[subsystem])
            end
            twin_index = kron_linear_index(interleaved, twin_dimensions)
            vectorized_index = ket_index + (bra_index - 1) * total
            permutation[twin_index, vectorized_index] = 1
        end
    end
    permutation
end
