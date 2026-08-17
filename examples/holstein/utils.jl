using LinearAlgebra

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
    site_projectors(site_count)

Return one projector onto each site in a site-local basis.
"""
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

"""
    HolsteinConfig(; kwargs...)

Store the physical and numerical settings for a periodic Holstein HEOM-TT run.
`brownian_damping_cm` is QFiND's Brownian damping parameter `Γ_Q`; its poles
decay at `Γ_Q / 2` (100 cm⁻¹ for the default `Γ_Q = 200 cm⁻¹`).
"""
struct HolsteinConfig
    site_count::Int
    site_energies_cm::Vector{Float64}
    hopping_cm::Float64
    brownian_frequency_cm::Float64
    # QFiND Brownian damping parameter Γ_Q; pole decay is Γ_Q / 2.
    brownian_damping_cm::Float64
    reorganization_energy_cm::Float64
    temperature_K::Float64
    initial_site::Int
    final_time_fs::Float64
    time_step_fs::Float64
    bcf_final_time_fs::Float64
    bcf_sample_count::Int
    bcf_fit_tolerance::Float64
    bcf_upper_bound_cm::Float64
    hierarchy_local_size::Int
    temporal_basis_size::Int
    tamen_tolerance::Float64
    operator_tolerance::Float64
    state_rounding_tolerance::Float64
    sweep_count::Int
    local_iterations::Int
    kick_rank::Int
    progress_interval::Int
end

function HolsteinConfig(;
    site_count=5,
    site_energies_cm=zeros(site_count),
    hopping_cm=400.0,
    brownian_frequency_cm=1400.0,
    brownian_damping_cm=200.0, # QFiND Γ_Q; default pole decay is 100 cm⁻¹
    reorganization_energy_cm=600.0,
    temperature_K=300.0,
    initial_site=1,
    final_time_fs=100.0,
    time_step_fs=1.0,
    bcf_final_time_fs=100.0,
    bcf_sample_count=400,
    bcf_fit_tolerance=1e-6,
    bcf_upper_bound_cm=10_000.0,
    hierarchy_local_size=4,
    temporal_basis_size=3,
    tamen_tolerance=1e-4,
    operator_tolerance=1e-10,
    state_rounding_tolerance=1e-10,
    sweep_count=5,
    local_iterations=10,
    kick_rank=4,
    progress_interval=10,
)
    config = HolsteinConfig(
        Int(site_count), Float64.(site_energies_cm), Float64(hopping_cm),
        Float64(brownian_frequency_cm), Float64(brownian_damping_cm),
        Float64(reorganization_energy_cm), Float64(temperature_K),
        Int(initial_site), Float64(final_time_fs), Float64(time_step_fs),
        Float64(bcf_final_time_fs), Int(bcf_sample_count),
        Float64(bcf_fit_tolerance), Float64(bcf_upper_bound_cm),
        Int(hierarchy_local_size), Int(temporal_basis_size),
        Float64(tamen_tolerance), Float64(operator_tolerance),
        Float64(state_rounding_tolerance), Int(sweep_count),
        Int(local_iterations), Int(kick_rank), Int(progress_interval),
    )
    return validate_config(config)
end

"""
    validate_config(config)

Validate physical and numerical settings and return `config` unchanged.
"""
function validate_config(config::HolsteinConfig)
    config.site_count >= 2 || throw(ArgumentError("site_count must be at least two"))
    length(config.site_energies_cm) == config.site_count ||
        throw(ArgumentError("site_energies_cm length must equal site_count"))
    1 <= config.initial_site <= config.site_count ||
        throw(ArgumentError("initial_site must index a site"))

    all(isfinite, config.site_energies_cm) ||
        throw(ArgumentError("site_energies_cm must be finite"))
    scalar_values = (
        config.hopping_cm,
        config.brownian_frequency_cm,
        config.brownian_damping_cm,
        config.reorganization_energy_cm,
        config.temperature_K,
        config.final_time_fs,
        config.time_step_fs,
        config.bcf_final_time_fs,
        config.bcf_fit_tolerance,
        config.bcf_upper_bound_cm,
        config.tamen_tolerance,
        config.operator_tolerance,
        config.state_rounding_tolerance,
    )
    all(isfinite, scalar_values) || throw(ArgumentError("configuration values must be finite"))

    config.hopping_cm >= 0 || throw(ArgumentError("hopping_cm must be nonnegative"))
    all(>(0), (
        config.brownian_frequency_cm,
        config.brownian_damping_cm,
        config.reorganization_energy_cm,
        config.temperature_K,
        config.final_time_fs,
        config.time_step_fs,
        config.bcf_final_time_fs,
        config.bcf_fit_tolerance,
        config.bcf_upper_bound_cm,
        config.tamen_tolerance,
        config.operator_tolerance,
        config.state_rounding_tolerance,
    )) || throw(ArgumentError("physical and numerical scales must be positive"))

    config.bcf_sample_count >= 2 ||
        throw(ArgumentError("bcf_sample_count must be at least two"))
    all(>(0), (
        config.hierarchy_local_size,
        config.temporal_basis_size,
        config.sweep_count,
        config.local_iterations,
        config.kick_rank,
        config.progress_interval,
    )) || throw(ArgumentError("solver size and iteration controls must be positive"))

    step_count = config.final_time_fs / config.time_step_fs
    isapprox(step_count, round(step_count); atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("final_time_fs must be an integer multiple of time_step_fs"))
    return config
end
