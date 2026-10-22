using LinearAlgebra
using TTDynamics
using TTSolver
using QFiND
using HEOMKit: icm2ifs, BathExp, NoiseExp

function bond_operators(site_count::Integer)
    site_count >= 3 || throw(ArgumentError("periodic Peierls chain needs at least three sites"))
    symmetric = Matrix{ComplexF64}[]
    antisymmetric = Matrix{ComplexF64}[]
    for site in 1:site_count
        neighbor = mod1(site + 1, site_count)
        bond = zeros(ComplexF64, site_count, site_count)
        difference = zeros(ComplexF64, site_count, site_count)
        bond[neighbor, site] = bond[site, neighbor] = 1
        difference[neighbor, site] = 1
        difference[site, neighbor] = -1
        push!(symmetric, bond)
        push!(antisymmetric, difference)
    end
    return (; symmetric, antisymmetric)
end

function peierls_hamiltonian(config)
    length(config.site_energies_cm) == config.site_count ||
        throw(DimensionMismatch("site_energies_cm length must equal site_count"))
    bonds = bond_operators(config.site_count)
    hamiltonian = Matrix(Diagonal(ComplexF64.(config.site_energies_cm)))
    for bond in bonds.symmetric
        hamiltonian -= config.hopping_cm * bond
    end
    return hamiltonian
end

function peierls_couplings(site_count::Integer)
    bonds = bond_operators(site_count).symmetric
    return [bonds[site] - bonds[mod1(site - 1, site_count)]
            for site in 1:site_count]
end

function decompose_bath(config)
    config.drude_cutoff_cm > 0 || throw(ArgumentError("drude_cutoff_cm must be positive"))
    config.reorganization_energy_cm > 0 ||
        throw(ArgumentError("reorganization_energy_cm must be positive"))
    spectral_density = DrudeSD(config.drude_cutoff_cm,
                               config.reorganization_energy_cm)
    exponents, coefficients = if config.decomposition_method === :tpsd
        tpsd(spectral_density, config.temperature_K, config.pade_order,
             config.tpsd_tolerance; pade_type=config.pade_type)
    elseif config.decomposition_method === :psd
        psd(spectral_density, config.temperature_K, config.pade_order;
            pade_type=config.pade_type)
    else
        throw(ArgumentError("decomposition_method must be :tpsd or :psd"))
    end
    exponents = ComplexF64.(exponents)
    coefficients = ComplexF64.(coefficients)
    isempty(exponents) && error("Drude decomposition returned no exponential terms")
    all(isfinite, exponents) && all(>(0), real.(exponents)) ||
        error("Drude decomposition returned invalid exponents")
    all(isfinite, coefficients) || error("Drude decomposition returned invalid coefficients")
    return (; exponents, coefficients)
end

function build_problem(config, decomposition)
    hamiltonian = peierls_hamiltonian(config) * icm2ifs
    baths = [BathExp(decomposition.exponents, decomposition.coefficients, coupling)
             for coupling in peierls_couplings(config.site_count)]
    system = HEOMTTSystem(hamiltonian, NoiseExp(baths),
                          config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable, population_observables)
end

function peierls_current_action(config, system::HEOMTTSystem)
    config.site_count == size(system.H_sys, 1) ||
        throw(DimensionMismatch("system and config site counts differ"))
    system.noise.nbath == config.site_count ||
        throw(DimensionMismatch("one bath per site is required"))
    local_currents = [-im * difference for difference in
                      bond_operators(config.site_count).antisymmetric]
    coordinates = [build_heom_bath_coordinate_left_action(
        system, site; tol=config.operator_tolerance,
    ) for site in 1:config.site_count]
    result = nothing
    for site in 1:config.site_count
        current = build_heom_left_action(system, local_currents[site];
                                         tol=config.operator_tolerance)
        term = (-config.hopping_cm * icm2ifs) * current +
               current * (coordinates[site] - coordinates[mod1(site + 1,
                                                                config.site_count)])
        result = isnothing(result) ? term : result + term
        result = tt_round(result, config.operator_tolerance)
    end
    return result
end
