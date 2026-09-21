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
    function ExponentialCorrelation(rates::Vector{ComplexF64}, coeff_forward::Vector{ComplexF64},
                                    coeff_backward::Vector{ComplexF64}, scales::Vector{Float64},
                                    metadata::CorrelationFitMetadata)
        _validate_exponential_correlation(rates, coeff_forward, coeff_backward, scales; stability_tolerance=0.0)
        new(rates, coeff_forward, coeff_backward, scales, metadata)
    end
end

function ExponentialCorrelation(rates, forward, backward, metadata::CorrelationFitMetadata;
                                scale_epsilon=eps(Float64), stability_tolerance=1e-12)
    isfinite(scale_epsilon) && scale_epsilon > 0 || throw(ArgumentError("scale_epsilon must be finite and positive"))
    isfinite(stability_tolerance) && stability_tolerance >= 0 || throw(ArgumentError("stability_tolerance must be finite and nonnegative"))
    rates_fixed = ComplexF64.(rates)
    coeff_forward = ComplexF64.(forward)
    coeff_backward = ComplexF64.(backward)
    all(isfinite, rates_fixed) || throw(ArgumentError("rates must be finite"))
    all(isfinite, coeff_forward) || throw(ArgumentError("forward coefficients must be finite"))
    all(isfinite, coeff_backward) || throw(ArgumentError("backward coefficients must be finite"))
    _validate_matching_nonempty_lengths(rates_fixed, coeff_forward, coeff_backward)
    for index in eachindex(rates_fixed)
        real_part = real(rates_fixed[index])
        real_part < -stability_tolerance && throw(ArgumentError("rate $index has negative real part $real_part beyond tolerance $stability_tolerance"))
        real_part < 0 && (rates_fixed[index] = complex(0.0, imag(rates_fixed[index])))
    end
    scales = sqrt.(max.(abs.(coeff_forward), abs.(coeff_backward), scale_epsilon))
    ExponentialCorrelation(rates_fixed, coeff_forward, coeff_backward, scales, metadata)
end

function _validate_exponential_correlation(rates, coeff_forward, coeff_backward, scales; stability_tolerance::Float64=0.0)
    _validate_matching_nonempty_lengths(rates, coeff_forward, coeff_backward)
    all(isfinite, rates) && all(isfinite, coeff_forward) && all(isfinite, coeff_backward) && all(isfinite, scales) ||
        throw(ArgumentError("correlation data must be finite"))
    all(>(0), scales) || throw(ArgumentError("scales must be positive"))
    for (index, rate) in pairs(rates)
        real(rate) < -stability_tolerance && throw(ArgumentError("rate $index has negative real part $(real(rate)) beyond tolerance $stability_tolerance"))
    end
end

function _validate_matching_nonempty_lengths(vectors...)
    lengths = length.(vectors)
    all(>(0), lengths) || throw(ArgumentError("correlation data must be nonempty"))
    all(==(first(lengths)), lengths) || throw(ArgumentError("correlation data must have equal lengths"))
end

function _validate_local_operator(operator::AbstractMatrix)
    size(operator, 1) == size(operator, 2) && !isempty(operator) || throw(ArgumentError("local bath operator must be a nonempty square matrix"))
    all(isfinite, operator) || throw(ArgumentError("local bath operator must be finite"))
end

_complex_tt_matrix(matrix::TTMatrix) = TTMatrix([ComplexF64.(core) for core in matrix.cores])
