using LinearAlgebra

function periodic_lattice_distance(m::Integer, n::Integer, site_count::Integer)::Int
    site_count >= 2 || throw(ArgumentError("site_count must be at least two"))
    1 <= m <= site_count || throw(ArgumentError("m must index a lattice site"))
    1 <= n <= site_count || throw(ArgumentError("n must index a lattice site"))
    separation = abs(Int(m) - Int(n))
    return min(separation, Int(site_count) - separation)
end

function default_frohlich_kernel(distance::Integer)::Float64
    distance >= 0 || throw(ArgumentError("distance must be nonnegative"))
    return (Float64(distance)^2 + 1.0)^(-1.5)
end

function _is_circulant(matrix::AbstractMatrix)
    row_count, column_count = size(matrix)
    row_count == column_count || return false
    return all(1:column_count) do column
        next_column = mod1(column + 1, column_count)
        isapprox(
            @view(matrix[:, next_column]),
            circshift(@view(matrix[:, column]), 1);
            rtol=1e-12,
            atol=1e-14,
        )
    end
end

function normalized_frohlich_kernel(site_count::Integer;
                                     kernel=default_frohlich_kernel)::Matrix{Float64}
    site_count >= 2 || throw(ArgumentError("site_count must be at least two"))
    count = Int(site_count)
    raw = Matrix{Float64}(undef, count, count)
    for bath_site in 1:count, electronic_site in 1:count
        value = Float64(kernel(periodic_lattice_distance(bath_site, electronic_site, count)))
        isfinite(value) || throw(ArgumentError("kernel values must be finite"))
        value >= 0 || throw(ArgumentError("kernel values must be nonnegative"))
        raw[bath_site, electronic_site] = value
    end
    isapprox(raw, transpose(raw); rtol=1e-12, atol=1e-14) ||
        error("raw Frohlich kernel violates lattice symmetry")
    _is_circulant(raw) || error("raw Frohlich kernel violates lattice translation symmetry")

    weights = similar(raw)
    for electronic_site in 1:count
        scale = sqrt(sum(abs2, @view raw[:, electronic_site]))
        isfinite(scale) && scale > 0 ||
            throw(ArgumentError("each electronic site must have nonzero finite coupling"))
        weights[:, electronic_site] .= raw[:, electronic_site] ./ scale
    end
    all(n -> isapprox(sum(abs2, @view weights[:, n]), 1.0; atol=1e-14), 1:count) ||
        error("normalized Frohlich kernel violates unit column norm")
    isapprox(weights, transpose(weights); rtol=1e-12, atol=1e-14) ||
        error("normalized Frohlich kernel violates lattice symmetry")
    _is_circulant(weights) ||
        error("normalized Frohlich kernel violates lattice translation symmetry")
    return weights
end

function frohlich_coupling_operators(site_count::Integer;
                                     kernel=default_frohlich_kernel)
    weights = normalized_frohlich_kernel(site_count; kernel)
    operators = [Matrix(Diagonal(ComplexF64.(weights[m, :]))) for m in axes(weights, 1)]
    all(isdiag, operators) || error("Frohlich coupling operators must be diagonal")
    all(ishermitian, operators) || error("Frohlich coupling operators must be Hermitian")
    count = length(operators)
    all(1:count) do bath_site
        next_bath_site = mod1(bath_site + 1, count)
        isapprox(
            diag(operators[next_bath_site]),
            circshift(diag(operators[bath_site]), 1);
            rtol=1e-12,
            atol=1e-14,
        )
    end || error("Frohlich coupling operators violate lattice translation symmetry")
    return operators
end
