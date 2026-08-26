using LinearAlgebra
using TTDynamics
using TTSolver

struct QuarticModel
    config::QuarticConfig
    mode::QuarticMode
    fit::Union{Nothing,ExponentialCorrelation}
    system_hamiltonian::TTMatrix{ComplexF64}
    system::MultiCoreHEOMTTSystem
    generator::TTMatrix{ComplexF64}
    initial_state::TTTensor{ComplexF64}
    observables::NamedTuple
end

function _quartic_electron_hamiltonian(config::QuarticConfig)
    hamiltonian = Matrix(Diagonal(ComplexF64.(config.site_energies)))
    for site in 1:(config.site_count - 1)
        hamiltonian[site, site + 1] = -config.hopping
        hamiltonian[site + 1, site] = -config.hopping
    end
    return hamiltonian
end

function _quartic_site_projector(site_count::Int, site::Int)
    1 <= site <= site_count || throw(ArgumentError(
        "site must be in 1:$site_count",
    ))
    projector = zeros(ComplexF64, site_count, site_count)
    projector[site, site] = 1
    return projector
end

function _quartic_hamiltonian_terms(config::QuarticConfig, mode::QuarticMode)
    size(mode.hamiltonian) == (config.d_keep, config.d_keep) || throw(ArgumentError(
        "mode dimension must equal d_keep",
    ))
    electron_identity = Matrix{ComplexF64}(I, config.site_count, config.site_count)
    oscillator_identity = Matrix{ComplexF64}(I, config.d_keep, config.d_keep)
    base_factors = Matrix{ComplexF64}[
        electron_identity,
        [copy(oscillator_identity) for _ in 1:config.site_count]...,
    ]
    terms = Vector{Vector{Matrix{ComplexF64}}}()

    electron_factors = copy(base_factors)
    electron_factors[1] = _quartic_electron_hamiltonian(config)
    push!(terms, electron_factors)

    for site in 1:config.site_count
        oscillator_factors = copy(base_factors)
        oscillator_factors[site + 1] = mode.hamiltonian
        push!(terms, oscillator_factors)

        if !iszero(config.g)
            coupling_factors = copy(base_factors)
            coupling_factors[1] = config.g .* _quartic_site_projector(config.site_count, site)
            coupling_factors[site + 1] = mode.q
            push!(terms, coupling_factors)
        end
    end
    return terms
end

function _quartic_sum_tt_terms(terms, tolerance)
    result = tt_mkron(first(terms))
    for factors in Iterators.drop(terms, 1)
        result = tt_round(result + tt_mkron(factors), tolerance)
    end
    return tt_round(result, tolerance)
end

"""
    build_system_mpo(config, mode)

Build the electron--quartic-oscillator Hamiltonian over one electronic Hilbert
core followed by one Hilbert core per oscillator. Bath parameters do not enter
this operator; no bath counter term is included.
"""
function build_system_mpo(config::QuarticConfig, mode::QuarticMode)
    validate_quartic_config(config)
    terms = _quartic_hamiltonian_terms(config, mode)
    return _quartic_sum_tt_terms(terms, config.operator_tolerance)
end

function _quartic_physical_liouvillian(config::QuarticConfig, mode::QuarticMode)
    terms = _quartic_hamiltonian_terms(config, mode)
    first_factors = first(terms)
    liouvillian = -1im * (
        tt_mkron(liouville_left.(first_factors)) -
        tt_mkron(liouville_right.(first_factors))
    )
    for factors in Iterators.drop(terms, 1)
        term = -1im * (
            tt_mkron(liouville_left.(factors)) -
            tt_mkron(liouville_right.(factors))
        )
        liouvillian = tt_round(liouvillian + term, config.operator_tolerance)
    end
    return tt_round(liouvillian, config.operator_tolerance)
end

function _quartic_observable(factors)
    cores = [
        reshape(vec(ComplexF64.(factor)), 1, length(factor), 1)
        for factor in factors
    ]
    return TTTensor(cores)
end

"""
    quartic_observables(config, mode)

Build physical-only TT vectors for root-ADO traces, electron populations and
number, and each oscillator's position, squared position, and bare energy.
"""
function quartic_observables(config::QuarticConfig, mode::QuarticMode)
    validate_quartic_config(config)
    size(mode.hamiltonian) == (config.d_keep, config.d_keep) || throw(ArgumentError(
        "mode dimension must equal d_keep",
    ))
    electron_identity = Matrix{ComplexF64}(I, config.site_count, config.site_count)
    oscillator_identity = Matrix{ComplexF64}(I, config.d_keep, config.d_keep)
    identities = Matrix{ComplexF64}[
        electron_identity,
        [copy(oscillator_identity) for _ in 1:config.site_count]...,
    ]

    electron_populations = TTTensor{ComplexF64}[]
    for site in 1:config.site_count
        factors = copy(identities)
        factors[1] = _quartic_site_projector(config.site_count, site)
        push!(electron_populations, _quartic_observable(factors))
    end

    oscillator_q = TTTensor{ComplexF64}[]
    oscillator_q2 = TTTensor{ComplexF64}[]
    oscillator_hamiltonian = TTTensor{ComplexF64}[]
    for site in 1:config.site_count
        q_factors = copy(identities)
        q_factors[site + 1] = mode.q
        push!(oscillator_q, _quartic_observable(q_factors))

        q2_factors = copy(identities)
        q2_factors[site + 1] = mode.q2
        push!(oscillator_q2, _quartic_observable(q2_factors))

        hamiltonian_factors = copy(identities)
        hamiltonian_factors[site + 1] = mode.hamiltonian
        push!(oscillator_hamiltonian, _quartic_observable(hamiltonian_factors))
    end

    trace_observable = _quartic_observable(identities)
    electron_number = first(electron_populations)
    for population in Iterators.drop(electron_populations, 1)
        electron_number = tt_round(
            electron_number + population,
            config.operator_tolerance,
        )
    end
    return (;
        trace=trace_observable,
        electron_populations,
        electron_number,
        oscillator_q,
        oscillator_q2,
        oscillator_hamiltonian,
    )
end

"""
    build_factorized_initial_tt(config, mode, system; initial_site=1)

Build a localized electronic density, one bare thermal oscillator density per
site, and hierarchy vacuum occupations.
"""
function build_factorized_initial_tt(
    config::QuarticConfig,
    mode::QuarticMode,
    system::MultiCoreHEOMTTSystem;
    initial_site::Integer=1,
)
    site = Int(initial_site)
    electron_density = _quartic_site_projector(config.site_count, site)
    oscillator_density = quartic_thermal_density(mode, config.temperature)
    physical_densities = Matrix{ComplexF64}[
        electron_density,
        [copy(oscillator_density) for _ in 1:config.site_count]...,
    ]
    return build_multicore_heom_initial_state(
        system,
        physical_densities;
        tol=config.state_rounding_tolerance,
    )
end

build_factorized_initial_tt(
    system::MultiCoreHEOMTTSystem,
    config::QuarticConfig,
    mode::QuarticMode;
    kwargs...,
) = build_factorized_initial_tt(config, mode, system; kwargs...)

"""
    build_quartic_model(config, mode, fit=nothing; initial_site=1)

Construct the physical Hamiltonian and Liouvillian, optional site-local HEOM
baths, generator, factorized initial state, and physical observables. A zero
`bath_lambda` selects the generic closed-system path and creates no hierarchy
cores, so no artificial zero correlation scales are introduced.
"""
function build_quartic_model(
    config::QuarticConfig,
    mode::QuarticMode,
    fit::Union{Nothing,ExponentialCorrelation}=nothing;
    initial_site::Integer=1,
)
    validate_quartic_config(config)
    system_hamiltonian = build_system_mpo(config, mode)
    physical_liouvillian = _quartic_physical_liouvillian(config, mode)
    physical_dimensions = [config.site_count^2; fill(config.d_keep^2, config.site_count)]

    active_fit = if iszero(config.bath_lambda)
        nothing
    else
        isnothing(fit) && throw(ArgumentError(
            "a fitted bath correlation is required when bath_lambda is positive",
        ))
        fit
    end
    couplings = if isnothing(active_fit)
        LocalBathCoupling[]
    else
        [
            LocalBathCoupling(site + 1, mode.q, active_fit)
            for site in 1:config.site_count
        ]
    end
    hierarchy_sizes = if isnothing(active_fit)
        Int[]
    else
        fill(
            config.hierarchy_nmax + 1,
            config.site_count * length(active_fit.rates),
        )
    end
    system = MultiCoreHEOMTTSystem(
        physical_liouvillian,
        physical_dimensions,
        couplings,
        hierarchy_sizes,
    )
    generator = build_multicore_heom_generator(
        system;
        tol=config.operator_tolerance,
    )
    initial_state = build_factorized_initial_tt(
        config,
        mode,
        system;
        initial_site=initial_site,
    )
    observables = quartic_observables(config, mode)
    return QuarticModel(
        config,
        mode,
        active_fit,
        system_hamiltonian,
        system,
        generator,
        initial_state,
        observables,
    )
end
