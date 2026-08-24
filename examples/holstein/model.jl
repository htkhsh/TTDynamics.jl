"""
    periodic_holstein_hamiltonian(site_energies, hopping)

Construct the single-excitation Hamiltonian for a periodic Holstein chain.
"""
function periodic_holstein_hamiltonian(site_energies::AbstractVector,
                                       hopping::Real)
    site_count = length(site_energies)
    site_count >= 2 || throw(ArgumentError("site_energies must contain at least two sites"))
    hopping >= 0 || throw(ArgumentError("hopping must be nonnegative"))

    H = Matrix(Diagonal(ComplexF64.(site_energies)))
    for site in 1:(site_count - 1)
        H[site, site + 1] = H[site + 1, site] = -hopping
    end
    if site_count > 2
        H[site_count, 1] = H[1, site_count] = -hopping
    end
    return H
end

"""
    holstein_site_projectors(site_count)

Return one projector onto each site in a site-local basis.
"""
function holstein_site_projectors(site_count::Integer)
    site_count >= 2 || throw(ArgumentError("site_count must be at least two"))
    projectors = Matrix{ComplexF64}[]
    for site in 1:site_count
        projector = zeros(ComplexF64, site_count, site_count)
        projector[site, site] = 1
        push!(projectors, projector)
    end
    return projectors
end

"""
    decompose_holstein_bath(config)

Decompose the finite-temperature Brownian bath correlation function described
by `config` with QFiND TPSD. Its damping field is passed directly as QFiND's
`Γ_Q`, whose corresponding pole decay is `Γ_Q / 2`.
"""
function decompose_holstein_bath(config::HolsteinConfig)
    spectral_density = BrownianSD(
        config.brownian_frequency_cm,
        config.brownian_damping_cm,
        config.reorganization_energy_cm,
    )
    exponents, coefficients = tpsd(
        spectral_density,
        config.temperature_K,
        config.pade_order,
        config.tpsd_tolerance;
        pade_type=config.pade_type,
    )
    exponents = ComplexF64.(exponents)
    coefficients = ComplexF64.(coefficients)
    isempty(exponents) && error("Brownian TPSD returned no exponential terms")
    all(isfinite, exponents) || error("Brownian TPSD returned nonfinite exponents")
    all(isfinite, coefficients) || error("Brownian TPSD returned nonfinite coefficients")
    # TPSD and BathExp use C(t) = sum(c[k] * exp(-exponents[k] * t)),
    # hence a decaying expansion has strictly positive real decay rates.
    all(>(0), real.(exponents)) ||
        error("Brownian TPSD returned a nondecaying exponential")

    reference_bcf = BosonicBCF(
        spectral_density,
        config.temperature_K;
        ub=config.bcf_upper_bound_cm,
    )
    validation_times = collect(range(
        0.0,
        config.validation_final_time_fs;
        length=config.validation_sample_count,
    ))
    reference_samples = reference_bcf.(validation_times)
    tpsd_samples = bcf_approx(validation_times, exponents, coefficients)
    reference_norm = norm(reference_samples)
    reference_norm > 0 || error("Brownian BCF validation norm is zero")
    relative_error = norm(tpsd_samples - reference_samples) / reference_norm
    isfinite(relative_error) || error("Brownian TPSD validation error is nonfinite")

    return (;
        exponents,
        coefficients,
        relative_error,
        validation_times,
        reference_samples,
        tpsd_samples,
    )
end

"""
    build_holstein_model(config, decomposition)

Construct the periodic Holstein Hamiltonian and its independent site-local
Brownian baths in the HEOM-TT representation.
"""
function build_holstein_model(config::HolsteinConfig, decomposition)
    H_cm = periodic_holstein_hamiltonian(
        config.site_energies_cm,
        config.hopping_cm,
    )
    H_fs = H_cm * icm2ifs
    baths = [BathExp(decomposition.exponents, decomposition.coefficients, projector)
             for projector in holstein_site_projectors(config.site_count)]
    system = HEOMTTSystem(H_fs, NoiseExp(baths), config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)

    initial_state = build_initial_state(
        system,
        config.initial_site;
        tol=config.operator_tolerance,
    )
    tt_dimensions = heom_tt_dimensions(system)
    println("  TPSD terms per site: $(length(decomposition.exponents))")
    println("  TT mode dimensions [ket, bra, hierarchy...]: $tt_dimensions")
    println("  Total TT cores: $(length(tt_dimensions)) (2 system cores + $(length(system.nb)) hierarchy cores)")
    println("  Hierarchy local sizes: $(system.nb)")
    println("  Initial TT ranks: $(tt_ranks(initial_state))")

    return (; system, liouvillian, trace_observable, population_observables)
end
