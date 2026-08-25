export build_multicore_heom_generator,
       build_multicore_heom_initial_state,
       root_ado,
       root_expectation

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

function _multicore_local_matrix(system::MultiCoreHEOMTTSystem, replacements...)
    dimensions = multicore_heom_dimensions(system)
    factors = [Matrix{ComplexF64}(I, dimension, dimension) for dimension in dimensions]
    for (index, factor) in replacements
        checkbounds(Bool, factors, index) || throw(ArgumentError(
            "multi-core HEOM local replacement index $index is out of range",
        ))
        size(factor) == size(factors[index]) || throw(ArgumentError(
            "multi-core HEOM local replacement at core $index must have dimensions " *
            "$(size(factors[index])); got $(size(factor))",
        ))
        factors[index] = ComplexF64.(factor)
    end
    return _local_product_matrix(factors)
end

function _extend_physical_liouvillian(system::MultiCoreHEOMTTSystem)
    cores = [copy(core) for core in system.physical_liouvillian.cores]
    for dimension in system.hierarchy_sizes
        identity_core = Matrix{ComplexF64}(I, dimension, dimension)
        push!(cores, reshape(identity_core, 1, dimension, dimension, 1))
    end
    return TTMatrix(cores)
end

"""    build_multicore_heom_generator(system; tol=1e-12) -> TTMatrix

Build the scaled multi-core HEOM generator without expanding a global physical
or hierarchy operator.
"""
function build_multicore_heom_generator(
    system::MultiCoreHEOMTTSystem;
    tol::Float64=1e-12,
)::TTMatrix
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))

    generator = _extend_physical_liouvillian(system)
    physical_core_count = length(system.physical_dimensions)
    hierarchy_index = 0

    for coupling in system.couplings
        physical_core = coupling.physical_core
        left_action = liouville_left(coupling.operator)
        right_action = liouville_right(coupling.operator)
        correlation = coupling.correlation

        for pole in eachindex(correlation.rates)
            hierarchy_index += 1
            hierarchy_core = physical_core_count + hierarchy_index
            dimension = system.hierarchy_sizes[hierarchy_index]
            number = diagm(0 => ComplexF64.(0:(dimension - 1)))
            lower = diagm(1 => sqrt.(ComplexF64.(1:(dimension - 1))))
            raise = diagm(-1 => sqrt.(ComplexF64.(1:(dimension - 1))))

            decay = _multicore_local_matrix(
                system,
                hierarchy_core => number,
            )
            upward = _multicore_local_matrix(
                system,
                physical_core => left_action - right_action,
                hierarchy_core => lower,
            )
            downward = _multicore_local_matrix(
                system,
                physical_core => (
                    correlation.coeff_forward[pole] * left_action -
                    correlation.coeff_backward[pole] * right_action
                ),
                hierarchy_core => raise,
            )

            generator = generator - correlation.rates[pole] * decay
            generator = generator - 1im * correlation.scales[pole] * upward
            generator = generator - 1im / correlation.scales[pole] * downward
            generator = tt_round(generator, tol)
        end
    end

    return tt_round(generator, tol)
end

"""    build_multicore_heom_initial_state(system, physical_density_cores; tol=1e-12)

Build a product of vectorized local physical density matrices and hierarchy
vacua.
"""
function build_multicore_heom_initial_state(
    system::MultiCoreHEOMTTSystem,
    physical_density_cores;
    tol::Float64=1e-12,
)::TTTensor
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
    length(physical_density_cores) == length(system.physical_dimensions) ||
        throw(ArgumentError(
            "physical_density_cores must contain one matrix per physical core",
        ))

    physical_vectors = Vector{Vector{ComplexF64}}(undef, length(physical_density_cores))
    for (index, density) in pairs(physical_density_cores)
        density isa AbstractMatrix || throw(ArgumentError(
            "physical density core $index must be a matrix",
        ))
        size(density, 1) == size(density, 2) || throw(ArgumentError(
            "physical density core $index must be square",
        ))
        length(density) == system.physical_dimensions[index] || throw(ArgumentError(
            "vectorized physical density core $index must have dimension " *
            "$(system.physical_dimensions[index])",
        ))
        all(isfinite, density) || throw(ArgumentError(
            "physical density core $index must be finite",
        ))
        physical_vectors[index] = vec(ComplexF64.(density))
    end

    hierarchy_vacua = Vector{Vector{ComplexF64}}(undef, length(system.hierarchy_sizes))
    for (index, dimension) in pairs(system.hierarchy_sizes)
        vacuum = zeros(ComplexF64, dimension)
        vacuum[1] = 1
        hierarchy_vacua[index] = vacuum
    end
    return tt_round(_local_product_tensor([physical_vectors; hierarchy_vacua]), tol)
end

function _validate_multicore_heom_state(
    state::TTTensor,
    system::MultiCoreHEOMTTSystem,
)
    expected = multicore_heom_dimensions(system)
    actual = tt_dims(state)
    actual == expected || throw(ArgumentError(
        "multi-core HEOM state dimensions must be $expected; got $actual",
    ))
    return nothing
end

"""    root_ado(state, system) -> TTTensor

Fix every hierarchy core at occupation zero and return the physical TT tensor.
"""
function root_ado(
    state::TTTensor,
    system::MultiCoreHEOMTTSystem,
)::TTTensor
    _validate_multicore_heom_state(state, system)
    physical_core_count = length(system.physical_dimensions)
    environment = ones(ComplexF64, 1)
    for core in Iterators.reverse(state.cores[(physical_core_count + 1):end])
        environment = ComplexF64.(core[:, 1, :]) * environment
    end

    physical_cores = [ComplexF64.(copy(core)) for core in state.cores[1:physical_core_count]]
    last_core = physical_cores[end]
    left_rank, dimension, right_rank = size(last_core)
    contracted = reshape(last_core, left_rank * dimension, right_rank) * environment
    physical_cores[end] = reshape(contracted, left_rank, dimension, 1)
    return TTTensor(physical_cores)
end

"""    root_expectation(state, system, observable)

Contract a physical observable TT vector with the root ADO without expanding
the hierarchy.
"""
function root_expectation(
    state::TTTensor,
    system::MultiCoreHEOMTTSystem,
    observable::TTTensor,
)
    tt_dims(observable) == system.physical_dimensions || throw(ArgumentError(
        "physical observable dimensions must be $(system.physical_dimensions); " *
        "got $(tt_dims(observable))",
    ))
    return tt_dot(observable, root_ado(state, system))
end

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
