using LinearAlgebra
using TTDynamics
using QFiND
using HEOMKit
using HEOMKit: icm2ifs

"""
    periodic_lattice_frohlich_hamiltonian(site_energies, hopping)

Construct the single-excitation Hamiltonian for a periodic lattice Fröhlich chain.
"""
function periodic_lattice_frohlich_hamiltonian(site_energies::AbstractVector,
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
    _is_circulant(raw) ||
        error("raw Frohlich kernel violates lattice translation symmetry")

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

"""
    decompose_lattice_frohlich_bath(config)

Decompose the finite-temperature Brownian bath correlation function described
by `config` with QFiND TPSD.
"""
function decompose_lattice_frohlich_bath(config::LatticeFrohlichConfig)
    validate_lattice_frohlich_config(config)
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

function build_lattice_frohlich_model(config::LatticeFrohlichConfig,
                                      decomposition;
                                      kernel=default_frohlich_kernel)
    validate_lattice_frohlich_config(config)
    H_fs = periodic_lattice_frohlich_hamiltonian(
        config.site_energies_cm, config.hopping_cm) * icm2ifs
    baths = [BathExp(decomposition.exponents, decomposition.coefficients, V)
             for V in frohlich_coupling_operators(config.site_count; kernel)]
    system = HEOMTTSystem(H_fs, NoiseExp(baths), config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable, population_observables)
end
