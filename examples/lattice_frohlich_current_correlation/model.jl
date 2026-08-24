using LinearAlgebra
using TTDynamics
using QFiND
using HEOMKit
using HEOMKit: icm2ifs

function periodic_lattice_frohlich_current_hamiltonian(
    site_energies::AbstractVector, hopping::Real,
)
    N = length(site_energies)
    N >= 2 || throw(ArgumentError("site_energies must contain at least two sites"))
    hopping >= 0 || throw(ArgumentError("hopping must be nonnegative"))

    hamiltonian = Matrix(Diagonal(ComplexF64.(site_energies)))
    for site in 1:(N - 1)
        hamiltonian[site, site + 1] = hamiltonian[site + 1, site] = -hopping
    end
    if N > 2
        hamiltonian[N, 1] = hamiltonian[1, N] = -hopping
    end
    return hamiltonian
end

function periodic_lattice_frohlich_distance(m::Integer, n::Integer, N::Integer)
    count, first_index, second_index = Int(N), Int(m), Int(n)
    count >= 2 || throw(ArgumentError("N must be at least two"))
    1 <= first_index <= count || throw(ArgumentError("m must index a lattice site"))
    1 <= second_index <= count || throw(ArgumentError("n must index a lattice site"))
    return min(abs(first_index - second_index), count - abs(first_index - second_index))
end

default_lattice_frohlich_current_kernel(distance::Integer) =
    (Float64(distance)^2 + 1.0)^(-1.5)

function _is_lattice_frohlich_current_circulant(matrix::AbstractMatrix)
    rows, columns = size(matrix)
    rows == columns || return false
    return all(1:columns) do column
        isapprox(
            @view(matrix[:, mod1(column + 1, columns)]),
            circshift(@view(matrix[:, column]), 1);
            rtol=1e-12,
            atol=1e-14,
        )
    end
end

function normalized_lattice_frohlich_current_kernel(
    N::Integer; kernel=default_lattice_frohlich_current_kernel,
)
    count = Int(N)
    count >= 2 || throw(ArgumentError("N must be at least two"))
    raw = Matrix{Float64}(undef, count, count)
    for m in 1:count, n in 1:count
        value = Float64(kernel(periodic_lattice_frohlich_distance(m, n, count)))
        isfinite(value) || throw(ArgumentError("kernel values must be finite"))
        value >= 0 || throw(ArgumentError("kernel values must be nonnegative"))
        raw[m, n] = value
    end
    isapprox(raw, transpose(raw); rtol=1e-12, atol=1e-14) ||
        error("raw Fröhlich kernel violates lattice symmetry")
    _is_lattice_frohlich_current_circulant(raw) ||
        error("raw Fröhlich kernel violates cyclic translation")

    weights = similar(raw)
    for n in 1:count
        scale = sqrt(sum(abs2, @view raw[:, n]))
        isfinite(scale) && scale > 0 ||
            throw(ArgumentError("each kernel column must have nonzero finite norm"))
        weights[:, n] .= raw[:, n] ./ scale
    end
    all(n -> isapprox(sum(abs2, @view weights[:, n]), 1.0; atol=1e-14), 1:count) ||
        error("normalized Fröhlich kernel violates unit column norm")
    isapprox(weights, transpose(weights); rtol=1e-12, atol=1e-14) ||
        error("normalized Fröhlich kernel violates lattice symmetry")
    _is_lattice_frohlich_current_circulant(weights) ||
        error("normalized Fröhlich kernel violates cyclic translation")
    return weights
end

function lattice_frohlich_current_coupling_operators(
    N::Integer; kernel=default_lattice_frohlich_current_kernel,
)
    weights = normalized_lattice_frohlich_current_kernel(N; kernel)
    operators = [Matrix(Diagonal(ComplexF64.(weights[m, :]))) for m in axes(weights, 1)]
    all(isdiag, operators) || error("Fröhlich coupling operators must be diagonal")
    all(ishermitian, operators) || error("Fröhlich coupling operators must be Hermitian")
    all(1:length(operators)) do m
        isapprox(
            diag(operators[mod1(m + 1, length(operators))]),
            circshift(diag(operators[m]), 1);
            rtol=1e-12,
            atol=1e-14,
        )
    end || error("Fröhlich coupling operators violate cyclic translation")
    return operators
end

function decompose_lattice_frohlich_current_bath(
    config::LatticeFrohlichCurrentCorrelationConfig,
)
    validate_lattice_frohlich_current_correlation_config(config)
    spectral_density = BrownianSD(
        config.brownian_frequency_cm,
        config.brownian_damping_cm,
        config.reorganization_energy_cm,
    )
    exponents, coefficients = tpsd(
        spectral_density, config.temperature_K, config.pade_order,
        config.tpsd_tolerance; pade_type=config.pade_type,
    )
    exponents, coefficients = ComplexF64.(exponents), ComplexF64.(coefficients)
    isempty(exponents) && error("Brownian TPSD returned no exponential terms")
    all(isfinite, exponents) || error("Brownian TPSD returned nonfinite exponents")
    all(isfinite, coefficients) || error("Brownian TPSD returned nonfinite coefficients")
    all(>(0), real.(exponents)) ||
        error("Brownian TPSD returned a nondecaying exponential")

    reference_bcf = BosonicBCF(
        spectral_density, config.temperature_K; ub=config.bcf_upper_bound_cm,
    )
    validation_times = collect(range(
        0.0, config.validation_final_time_fs; length=config.validation_sample_count,
    ))
    reference_samples = reference_bcf.(validation_times)
    tpsd_samples = bcf_approx(validation_times, exponents, coefficients)
    reference_norm = norm(reference_samples)
    reference_norm > 0 || error("Brownian BCF validation norm is zero")
    relative_error = norm(tpsd_samples - reference_samples) / reference_norm
    isfinite(relative_error) || error("Brownian TPSD validation error is nonfinite")
    return (; exponents, coefficients, relative_error, validation_times,
            reference_samples, tpsd_samples)
end

function build_lattice_frohlich_current_model(
    config::LatticeFrohlichCurrentCorrelationConfig,
    decomposition;
    kernel=default_lattice_frohlich_current_kernel,
)
    validate_lattice_frohlich_current_correlation_config(config)
    H = periodic_lattice_frohlich_current_hamiltonian(
        config.site_energies_cm, config.hopping_cm) * icm2ifs
    frohlich_weights = normalized_lattice_frohlich_current_kernel(
        config.site_count; kernel)
    couplings = lattice_frohlich_current_coupling_operators(config.site_count; kernel)
    baths = [BathExp(decomposition.exponents, decomposition.coefficients, V)
             for V in couplings]
    system = HEOMTTSystem(H, NoiseExp(baths), config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable,
            population_observables, frohlich_weights)
end
