using LinearAlgebra
using TTDynamics
using QFiND
using HEOMKit: BathExp, NoiseExp, icm2ifs

function lattice_frohlich_hamiltonian(config)
    n = config.site_count
    n >= 2 || throw(ArgumentError("site_count must be at least two"))
    length(config.site_energies_cm) == n ||
        throw(ArgumentError("site_energies_cm length must equal site_count"))
    hamiltonian = Matrix(Diagonal(ComplexF64.(config.site_energies_cm)))
    for site in 1:(n - 1)
        hamiltonian[site, site + 1] = hamiltonian[site + 1, site] = -config.hopping_cm
    end
    if n > 2
        hamiltonian[n, 1] = hamiltonian[1, n] = -config.hopping_cm
    end
    return hamiltonian
end

function lattice_distance(m::Integer, n::Integer, site_count::Integer)
    1 <= m <= site_count || throw(ArgumentError("m is out of range"))
    1 <= n <= site_count || throw(ArgumentError("n is out of range"))
    return min(abs(m - n), site_count - abs(m - n))
end

function frohlich_weights(site_count::Integer, interaction_range::Real)
    site_count >= 2 || throw(ArgumentError("site_count must be at least two"))
    isfinite(interaction_range) && interaction_range > 0 ||
        throw(ArgumentError("frohlich_range must be finite and positive"))
    weights = zeros(Float64, site_count, site_count)
    for site in 1:site_count, origin in 1:site_count
        distance = lattice_distance(site, origin, site_count)
        weights[site, origin] =
            (1 + (distance / interaction_range)^2)^(-1.5)
    end
    for origin in 1:site_count
        weights[:, origin] ./= norm(weights[:, origin])
    end
    return weights
end

"""Return the independent translated Fröhlich coupling operators S_l."""
function frohlich_coupling_operators(config)
    weights = frohlich_weights(config.site_count, config.frohlich_range)
    return [Matrix(Diagonal(ComplexF64.(weights[:, origin])))
            for origin in 1:config.site_count]
end

function current_operator(config)
    n = config.site_count
    scale = config.hopping_cm * icm2ifs
    current = zeros(ComplexF64, n, n)
    for site in 1:(n - 1)
        current[site + 1, site] += im * scale
        current[site, site + 1] -= im * scale
    end
    if n > 2
        current[1, n] += im * scale
        current[n, 1] -= im * scale
    end
    return current
end

function decompose_bath(config)
    spectral_density = BrownianSD(
        config.brownian_frequency_cm,
        config.brownian_damping_cm,
        config.reorganization_energy_cm,
    )
    exponents, coefficients = if config.decomposition_method === :tpsd
        tpsd(
            spectral_density,
            config.temperature_K,
            config.pade_order,
            config.tpsd_tolerance;
            pade_type=config.pade_type,
        )
    elseif config.decomposition_method === :psd
        psd(
            spectral_density,
            config.temperature_K,
            config.pade_order;
            pade_type=config.pade_type,
        )
    else
        throw(ArgumentError("decomposition_method must be :tpsd or :psd"))
    end
    exponents = ComplexF64.(exponents)
    coefficients = ComplexF64.(coefficients)
    isempty(exponents) && error("Brownian TPSD returned no exponential terms")
    all(isfinite, exponents) || error("Brownian TPSD returned nonfinite exponents")
    all(isfinite, coefficients) || error("Brownian TPSD returned nonfinite coefficients")
    all(>(0), real.(exponents)) || error("Brownian TPSD returned a nondecaying exponential")
    return (; exponents, coefficients)
end

function build_problem(config, decomposition)
    baths = [BathExp(decomposition.exponents, decomposition.coefficients, coupling)
             for coupling in frohlich_coupling_operators(config)]
    system = HEOMTTSystem(
        lattice_frohlich_hamiltonian(config) * icm2ifs,
        NoiseExp(baths),
        config.hierarchy_local_size,
    )
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable, population_observables)
end
