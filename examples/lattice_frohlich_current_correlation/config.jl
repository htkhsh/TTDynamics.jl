"""
    LatticeFrohlichCurrentCorrelationConfig(; kwargs...)

Store the physical and numerical settings for the periodic lattice Fröhlich
current-correlation workflow.
"""
struct LatticeFrohlichCurrentCorrelationConfig
    site_count::Int
    site_energies_cm::Vector{Float64}
    hopping_cm::Float64
    brownian_frequency_cm::Float64
    brownian_damping_cm::Float64
    reorganization_energy_cm::Float64
    temperature_K::Float64
    initial_site::Int
    final_time_fs::Float64
    time_step_fs::Float64
    pade_order::Int
    tpsd_tolerance::Float64
    pade_type::Symbol
    validation_final_time_fs::Float64
    validation_sample_count::Int
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

function LatticeFrohlichCurrentCorrelationConfig(
    ; site_count=5, site_energies_cm=zeros(site_count), hopping_cm=400.0,
    brownian_frequency_cm=1400.0, brownian_damping_cm=200.0,
    reorganization_energy_cm=600.0, temperature_K=300.0, initial_site=1,
    final_time_fs=500.0, time_step_fs=1.0, pade_order=8,
    tpsd_tolerance=2e-2, pade_type=:Nm1, validation_final_time_fs=100.0,
    validation_sample_count=200, bcf_upper_bound_cm=10_000.0,
    hierarchy_local_size=4, temporal_basis_size=3, tamen_tolerance=2e-2,
    operator_tolerance=1e-10, state_rounding_tolerance=1e-10,
    sweep_count=3, local_iterations=5, kick_rank=4, progress_interval=10,
)
    config = LatticeFrohlichCurrentCorrelationConfig(
        Int(site_count), Float64.(site_energies_cm), Float64(hopping_cm),
        Float64(brownian_frequency_cm), Float64(brownian_damping_cm),
        Float64(reorganization_energy_cm), Float64(temperature_K),
        Int(initial_site), Float64(final_time_fs), Float64(time_step_fs),
        Int(pade_order), Float64(tpsd_tolerance), Symbol(pade_type),
        Float64(validation_final_time_fs), Int(validation_sample_count),
        Float64(bcf_upper_bound_cm), Int(hierarchy_local_size),
        Int(temporal_basis_size), Float64(tamen_tolerance),
        Float64(operator_tolerance), Float64(state_rounding_tolerance),
        Int(sweep_count), Int(local_iterations), Int(kick_rank),
        Int(progress_interval),
    )
    return validate_lattice_frohlich_current_correlation_config(config)
end

"""
    validate_lattice_frohlich_current_correlation_config(config)

Validate physical and numerical settings and return `config` unchanged.
"""
function validate_lattice_frohlich_current_correlation_config(
    config::LatticeFrohlichCurrentCorrelationConfig,
)
    config.site_count >= 2 || throw(ArgumentError("site_count must be at least two"))
    length(config.site_energies_cm) == config.site_count ||
        throw(ArgumentError("site_energies_cm length must equal site_count"))
    all(isfinite, config.site_energies_cm) ||
        throw(ArgumentError("site_energies_cm must be finite"))
    1 <= config.initial_site <= config.site_count ||
        throw(ArgumentError("initial_site must index a site"))

    scalar_values = (
        config.hopping_cm, config.brownian_frequency_cm,
        config.brownian_damping_cm, config.reorganization_energy_cm,
        config.temperature_K, config.final_time_fs, config.time_step_fs,
        config.tpsd_tolerance, config.validation_final_time_fs,
        config.bcf_upper_bound_cm, config.tamen_tolerance,
        config.operator_tolerance, config.state_rounding_tolerance,
    )
    all(isfinite, scalar_values) ||
        throw(ArgumentError("configuration values must be finite"))
    config.hopping_cm >= 0 || throw(ArgumentError("hopping_cm must be nonnegative"))
    all(>(0), (
        config.brownian_frequency_cm, config.brownian_damping_cm,
        config.reorganization_energy_cm, config.temperature_K,
        config.final_time_fs, config.time_step_fs, config.tpsd_tolerance,
        config.validation_final_time_fs, config.bcf_upper_bound_cm,
        config.tamen_tolerance, config.operator_tolerance,
        config.state_rounding_tolerance,
    )) || throw(ArgumentError("physical and numerical scales must be positive"))
    config.brownian_damping_cm < 2 * config.brownian_frequency_cm ||
        throw(ArgumentError("brownian_damping_cm must give underdamped Brownian poles"))
    config.pade_order > 0 || throw(ArgumentError("pade_order must be positive"))
    config.pade_type in (:N, :Nm1) || throw(ArgumentError("pade_type must be :N or :Nm1"))
    config.validation_sample_count >= 2 ||
        throw(ArgumentError("validation_sample_count must be at least two"))
    all(>(0), (
        config.hierarchy_local_size, config.sweep_count, config.local_iterations,
        config.kick_rank, config.progress_interval,
    )) || throw(ArgumentError("solver size and iteration controls must be positive"))
    config.temporal_basis_size >= 3 && isodd(config.temporal_basis_size) ||
        throw(ArgumentError("temporal_basis_size must be odd and at least three"))

    step_count = config.final_time_fs / config.time_step_fs
    isapprox(step_count, round(step_count); atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("final_time_fs must be an integer multiple of time_step_fs"))
    return config
end
