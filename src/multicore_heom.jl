struct CorrelationFitMetadata
    training_absolute_error::Float64
    training_relative_error::Float64
    validation_absolute_error::Float64
    validation_relative_error::Float64
    sampling_interval::Float64
    fit_time_window::Tuple{Float64,Float64}
    singular_values::Vector{Float64}
    vandermonde_condition::Float64
    minimum_pole_separation::Float64
    maximum_coefficient_magnitude::Float64
    pole_corrections::Vector{String}
end

struct ExponentialCorrelation
    rates::Vector{ComplexF64}
    coeff_forward::Vector{ComplexF64}
    coeff_backward::Vector{ComplexF64}
    scales::Vector{Float64}
    metadata::CorrelationFitMetadata

    function ExponentialCorrelation(
        rates::Vector{ComplexF64},
        coeff_forward::Vector{ComplexF64},
        coeff_backward::Vector{ComplexF64},
        scales::Vector{Float64},
        metadata::CorrelationFitMetadata,
    )
        _validate_exponential_correlation(
            rates,
            coeff_forward,
            coeff_backward,
            scales;
            stability_tolerance=0.0,
        )
        new(rates, coeff_forward, coeff_backward, scales, metadata)
    end
end

function ExponentialCorrelation(
    rates,
    forward,
    backward,
    metadata::CorrelationFitMetadata;
    scale_epsilon=eps(Float64),
    stability_tolerance=1e-12,
)
    isfinite(scale_epsilon) && scale_epsilon > 0 ||
        throw(ArgumentError("scale_epsilon must be finite and positive"))
    isfinite(stability_tolerance) && stability_tolerance >= 0 ||
        throw(ArgumentError("stability_tolerance must be finite and nonnegative"))

    rates_fixed = ComplexF64.(rates)
    coeff_forward = ComplexF64.(forward)
    coeff_backward = ComplexF64.(backward)
    all(isfinite, rates_fixed) || throw(ArgumentError("rates must be finite"))
    all(isfinite, coeff_forward) || throw(ArgumentError("forward coefficients must be finite"))
    all(isfinite, coeff_backward) || throw(ArgumentError("backward coefficients must be finite"))
    _validate_matching_nonempty_lengths(rates_fixed, coeff_forward, coeff_backward)

    for index in eachindex(rates_fixed)
        real_part = real(rates_fixed[index])
        real_part < -stability_tolerance && throw(ArgumentError(
            "rate $index has negative real part $real_part beyond tolerance $stability_tolerance",
        ))
        if real_part < 0
            rates_fixed[index] = complex(0.0, imag(rates_fixed[index]))
        end
    end

    scales = sqrt.(max.(abs.(coeff_forward), abs.(coeff_backward), scale_epsilon))
    return ExponentialCorrelation(rates_fixed, coeff_forward, coeff_backward, scales, metadata)
end

struct LocalBathCoupling
    physical_core::Int
    operator::Matrix{ComplexF64}
    correlation::ExponentialCorrelation

    function LocalBathCoupling(
        physical_core::Int,
        operator::Matrix{ComplexF64},
        correlation::ExponentialCorrelation,
    )
        physical_core > 0 || throw(ArgumentError("physical_core must be positive"))
        _validate_local_operator(operator)
        new(physical_core, operator, correlation)
    end
end

function LocalBathCoupling(
    physical_core::Integer,
    operator::AbstractMatrix,
    correlation::ExponentialCorrelation,
)
    return LocalBathCoupling(Int(physical_core), ComplexF64.(operator), correlation)
end

LocalBathCoupling(coupling::LocalBathCoupling) = coupling

struct MultiCoreHEOMTTSystem
    physical_liouvillian::TTMatrix{ComplexF64}
    physical_dimensions::Vector{Int}
    couplings::Vector{LocalBathCoupling}
    hierarchy_sizes::Vector{Int}

    function MultiCoreHEOMTTSystem(
        physical_liouvillian::TTMatrix{ComplexF64},
        physical_dimensions::Vector{Int},
        couplings::Vector{LocalBathCoupling},
        hierarchy_sizes::Vector{Int},
    )
        _validate_multicore_heom_system(
            physical_liouvillian,
            physical_dimensions,
            couplings,
            hierarchy_sizes,
        )
        new(physical_liouvillian, physical_dimensions, couplings, hierarchy_sizes)
    end
end

function MultiCoreHEOMTTSystem(
    physical_liouvillian::TTMatrix,
    physical_dimensions,
    couplings,
    hierarchy_sizes,
)
    liouvillian = _complex_tt_matrix(physical_liouvillian)
    physical_dims = Int.(physical_dimensions)
    local_couplings = LocalBathCoupling.(couplings)
    hierarchy = Int.(hierarchy_sizes)

    _validate_multicore_heom_system(liouvillian, physical_dims, local_couplings, hierarchy)
    return MultiCoreHEOMTTSystem(liouvillian, physical_dims, local_couplings, hierarchy)
end

function liouville_left(A::AbstractMatrix)
    matrix = ComplexF64.(A)
    _validate_local_operator(matrix)
    identity_matrix = Matrix{ComplexF64}(I, size(matrix, 1), size(matrix, 1))
    return kron(identity_matrix, matrix)
end

function liouville_right(A::AbstractMatrix)
    matrix = ComplexF64.(A)
    _validate_local_operator(matrix)
    identity_matrix = Matrix{ComplexF64}(I, size(matrix, 1), size(matrix, 1))
    return kron(transpose(matrix), identity_matrix)
end

multicore_heom_dimensions(system::MultiCoreHEOMTTSystem) =
    [system.physical_dimensions; system.hierarchy_sizes]

function _validate_exponential_correlation(
    rates,
    coeff_forward,
    coeff_backward,
    scales;
    stability_tolerance::Float64=0.0,
)
    _validate_matching_nonempty_lengths(rates, coeff_forward, coeff_backward, scales)
    all(isfinite, rates) || throw(ArgumentError("rates must be finite"))
    all(isfinite, coeff_forward) || throw(ArgumentError("forward coefficients must be finite"))
    all(isfinite, coeff_backward) || throw(ArgumentError("backward coefficients must be finite"))
    all(isfinite, scales) || throw(ArgumentError("scales must be finite"))
    all(>(0), scales) || throw(ArgumentError("scales must be positive"))
    for (index, rate) in pairs(rates)
        real(rate) < -stability_tolerance && throw(ArgumentError(
            "rate $index has negative real part $(real(rate)) beyond tolerance $stability_tolerance",
        ))
    end
    return nothing
end

function _validate_matching_nonempty_lengths(vectors...)
    lengths = length.(vectors)
    all(>(0), lengths) || throw(ArgumentError("correlation data must be nonempty"))
    all(==(first(lengths)), lengths) ||
        throw(ArgumentError("correlation data must have equal lengths"))
    return nothing
end

function _validate_local_operator(operator::AbstractMatrix)
    size(operator, 1) == size(operator, 2) && !isempty(operator) ||
        throw(ArgumentError("local bath operator must be a nonempty square matrix"))
    all(isfinite, operator) ||
        throw(ArgumentError("local bath operator must be finite"))
    return nothing
end

function _complex_tt_matrix(matrix::TTMatrix)
    return TTMatrix([ComplexF64.(core) for core in matrix.cores])
end

function _validate_multicore_heom_system(
    physical_liouvillian::TTMatrix,
    physical_dimensions::Vector{Int},
    couplings::Vector{LocalBathCoupling},
    hierarchy_sizes::Vector{Int},
)
    isempty(physical_dimensions) &&
        throw(ArgumentError("physical_dimensions must be nonempty"))
    all(>(0), physical_dimensions) ||
        throw(ArgumentError("physical_dimensions must be positive"))
    all(>(0), hierarchy_sizes) ||
        throw(ArgumentError("hierarchy_sizes must be positive"))

    output_dims, input_dims = tt_dims(physical_liouvillian)
    output_dims == physical_dimensions ||
        throw(ArgumentError("physical Liouvillian output dimensions must match physical_dimensions"))
    input_dims == physical_dimensions ||
        throw(ArgumentError("physical Liouvillian input dimensions must match physical_dimensions"))

    expanded_poles = sum((length(coupling.correlation.rates) for coupling in couplings); init=0)
    length(hierarchy_sizes) == expanded_poles || throw(ArgumentError(
        "hierarchy_sizes must provide one positive local size per expanded bath pole",
    ))

    physical_core_count = length(physical_dimensions)
    for coupling in couplings
        1 <= coupling.physical_core <= physical_core_count || throw(ArgumentError(
            "physical_core $(coupling.physical_core) must be in 1:$physical_core_count",
        ))
        local_dim = size(coupling.operator, 1)
        expected_liouville_dim = local_dim^2
        physical_dimensions[coupling.physical_core] == expected_liouville_dim || throw(
            ArgumentError(
                "coupling core $(coupling.physical_core) expects Liouville dimension " *
                "$expected_liouville_dim from a $local_dim x $local_dim operator",
            ),
        )
    end

    return nothing
end
