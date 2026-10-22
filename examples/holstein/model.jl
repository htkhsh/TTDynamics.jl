using LinearAlgebra
using TTDynamics
using QFiND
using HEOMKit: icm2ifs, BathExp, NoiseExp

function holstein_hamiltonian(config)
    site_count = config.site_count
    site_count >= 2 || throw(ArgumentError("site_count must be at least two"))
    length(config.site_energies_cm) == site_count ||
        throw(ArgumentError("site_energies_cm length must equal site_count"))

    hamiltonian = Matrix(Diagonal(ComplexF64.(config.site_energies_cm)))
    for site in 1:(site_count - 1)
        hamiltonian[site, site + 1] = hamiltonian[site + 1, site] = -config.hopping_cm
    end
    if site_count > 2
        hamiltonian[site_count, 1] = hamiltonian[1, site_count] = -config.hopping_cm
    end
    return hamiltonian
end

function site_projectors(site_count::Integer)
    site_count >= 2 || throw(ArgumentError("site_count must be at least two"))
    projectors = Matrix{ComplexF64}[]
    for site in 1:site_count
        projector = zeros(ComplexF64, site_count, site_count)
        projector[site, site] = 1
        push!(projectors, projector)
    end
    return projectors
end

function current_operator(config)
    site_count = config.site_count
    scale = config.hopping_cm * icm2ifs
    current = zeros(ComplexF64, site_count, site_count)
    for site in 1:(site_count - 1)
        current[site + 1, site] += im * scale
        current[site, site + 1] -= im * scale
    end
    if site_count > 2
        current[1, site_count] += im * scale
        current[site_count, 1] -= im * scale
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
    all(>(0), real.(exponents)) ||
        error("Brownian TPSD returned a nondecaying exponential")
    return (; exponents, coefficients)
end

function build_problem(config, decomposition)
    hamiltonian = holstein_hamiltonian(config) * icm2ifs
    baths = [
        BathExp(decomposition.exponents, decomposition.coefficients, projector)
        for projector in site_projectors(config.site_count)
    ]
    system = HEOMTTSystem(
        hamiltonian,
        NoiseExp(baths),
        config.hierarchy_local_size,
    )
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable, population_observables)
end
