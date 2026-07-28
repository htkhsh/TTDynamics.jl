"""    franck_condon_factors(n, m, d) → Matrix{Float64}"""
function franck_condon_factors(n::Int, m::Int, d::Real)
    y = sqrt(2.0) / 2.0 .* [-d, d]
    F = zeros(Float64, n, m)
    
    F[1, 1] = exp(-d^2 / 4)
    for i in 1:n
        if i > 1
            F[i, 1] = y[1] * F[i-1, 1] / sqrt(i - 1)
        end
        for j in 2:m
            F[i, j] = y[2] * F[i, j-1]
            if i > 1
                F[i, j] += sqrt(i - 1) * F[i-1, j-1]
            end
            F[i, j] /= sqrt(j - 1)
        end
    end
    return F
end

"""    estimate_basis_size(freq, coupling; threshold=0.999, nb_min=10, nb_max=30) → Int"""
function estimate_basis_size(freq::Real, coupling::Real; 
                             threshold::Real=0.999, 
                             nb_min::Int=10, 
                             nb_max::Int=30)
    # Dimensionless displacement parameter
    shift = abs(freq) > 0 ? abs(coupling / freq) : 0.0
    
    if shift < 0.05
        # Small displacement: default basis is sufficient
        return nb_min
    end
    
    # Compute Franck-Condon factors from ground state
    FFC = franck_condon_factors(1, nb_max, shift)
    
    # Cumulative population: Σ|⟨0|n⟩|²
    cumsum_pop = cumsum(FFC[1, :].^2)
    
    # Find minimum basis size to capture threshold of population
    b0 = findfirst(x -> x >= threshold, cumsum_pop)
    if isnothing(b0)
        b0 = nb_max
    end
    
    # Take maximum of computed value and default
    return max(nb_min, min(b0, nb_max))
end

"""    estimate_basis_sizes(freqs, couplings; threshold=0.999, nb_min=10, nb_max=30) → Vector{Int}"""
function estimate_basis_sizes(freqs::AbstractVector{<:Real}, 
                              couplings::AbstractVector{<:Real}; 
                              threshold::Real=0.999, 
                              nb_min::Int=10, 
                              nb_max::Int=30)
    @assert length(freqs) == length(couplings) "freqs and couplings must have same length"
    
    return [estimate_basis_size(freqs[i], couplings[i]; 
                                threshold=threshold, 
                                nb_min=nb_min, 
                                nb_max=nb_max) 
            for i in eachindex(freqs)]
end
