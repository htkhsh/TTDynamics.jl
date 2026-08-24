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
    weights = similar(raw)
    for electronic_site in 1:count
        scale = sqrt(sum(abs2, @view raw[:, electronic_site]))
        isfinite(scale) && scale > 0 ||
            throw(ArgumentError("each electronic site must have nonzero finite coupling"))
        weights[:, electronic_site] .= raw[:, electronic_site] ./ scale
    end
    all(n -> isapprox(sum(abs2, @view weights[:, n]), 1.0; atol=1e-14), 1:count) ||
        error("normalized Frohlich kernel violates unit column norm")
    return weights
end

function frohlich_coupling_operators(site_count::Integer;
                                     kernel=default_frohlich_kernel)
    weights = normalized_frohlich_kernel(site_count; kernel)
    return [Matrix(Diagonal(ComplexF64.(weights[m, :]))) for m in axes(weights, 1)]
end
