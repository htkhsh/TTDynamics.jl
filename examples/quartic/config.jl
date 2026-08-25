"""
    QuarticConfig(; kwargs...)

Store physical, fit, hierarchy, and solver settings for quartic HEOM-TT
examples.
"""
struct QuarticConfig
    site_count::Int
    site_energies::Vector{Float64}
    hopping::Float64
    g::Float64
    basis_frequency::Float64
    d_raw::Int
    d_keep::Int
    omega_p::Float64
    K2::Float64
    K4::Float64
    temperature::Float64
    bath_lambda::Float64
    bath_gamma::Float64
    dt_fit::Float64
    t_fit_max::Float64
    validation_dt::Float64
    validation_t_max::Float64
    quadrature_atol::Float64
    quadrature_rtol::Float64
    quadrature_maxevals::Int
    fit_rank::Int
    fit_rank_max::Int
    fit_sval_rtol::Float64
    pole_stability_tolerance::Float64
    duplicate_pole_tolerance::Float64
    fit_absolute_tolerance::Float64
    fit_relative_tolerance::Float64
    hierarchy_nmax::Int
    final_time::Float64
    time_step::Float64
    temporal_basis_size::Int
    tamen_tolerance::Float64
    operator_tolerance::Float64
    state_rounding_tolerance::Float64
    sweep_count::Int
    local_iterations::Int
    kick_rank::Int
    tt_max_rank::Int
    trace_tolerance::Float64
    hermiticity_tolerance::Float64
    electron_number_tolerance::Float64
    progress_interval::Int
end

function QuarticConfig(; site_count=2, site_energies=zeros(site_count), hopping=1.0,
    g=1.0, basis_frequency=0.75, d_raw=16, d_keep=6, omega_p=2.0, K2=1.0,
    K4=0.1, temperature=1.0, bath_lambda=0.1, bath_gamma=1.0, dt_fit=0.05,
    t_fit_max=4.0, validation_dt=0.05, validation_t_max=6.0,
    quadrature_atol=1e-10, quadrature_rtol=1e-8, quadrature_maxevals=100_000,
    fit_rank=0, fit_rank_max=12, fit_sval_rtol=1e-10,
    pole_stability_tolerance=1e-10, duplicate_pole_tolerance=1e-8,
    fit_absolute_tolerance=1e-8, fit_relative_tolerance=1e-6,
    hierarchy_nmax=4, final_time=10.0, time_step=0.1, temporal_basis_size=3,
    tamen_tolerance=2e-2, operator_tolerance=1e-10,
    state_rounding_tolerance=1e-10, sweep_count=3, local_iterations=5,
    kick_rank=4, tt_max_rank=64, trace_tolerance=1e-8,
    hermiticity_tolerance=1e-8, electron_number_tolerance=1e-8,
    progress_interval=10)
    config = QuarticConfig(
        Int(site_count),
        Float64.(site_energies),
        Float64(hopping),
        Float64(g),
        Float64(basis_frequency),
        Int(d_raw),
        Int(d_keep),
        Float64(omega_p),
        Float64(K2),
        Float64(K4),
        Float64(temperature),
        Float64(bath_lambda),
        Float64(bath_gamma),
        Float64(dt_fit),
        Float64(t_fit_max),
        Float64(validation_dt),
        Float64(validation_t_max),
        Float64(quadrature_atol),
        Float64(quadrature_rtol),
        Int(quadrature_maxevals),
        Int(fit_rank),
        Int(fit_rank_max),
        Float64(fit_sval_rtol),
        Float64(pole_stability_tolerance),
        Float64(duplicate_pole_tolerance),
        Float64(fit_absolute_tolerance),
        Float64(fit_relative_tolerance),
        Int(hierarchy_nmax),
        Float64(final_time),
        Float64(time_step),
        Int(temporal_basis_size),
        Float64(tamen_tolerance),
        Float64(operator_tolerance),
        Float64(state_rounding_tolerance),
        Int(sweep_count),
        Int(local_iterations),
        Int(kick_rank),
        Int(tt_max_rank),
        Float64(trace_tolerance),
        Float64(hermiticity_tolerance),
        Float64(electron_number_tolerance),
        Int(progress_interval),
    )
    return validate_quartic_config(config)
end

"""
    validate_quartic_config(config)

Validate quartic example settings and return `config` unchanged.
"""
function validate_quartic_config(config::QuarticConfig)
    config.site_count >= 1 || throw(ArgumentError("site_count must be positive"))
    length(config.site_energies) == config.site_count ||
        throw(ArgumentError("site_energies length must equal site_count"))
    all(isfinite, config.site_energies) ||
        throw(ArgumentError("site_energies must be finite"))

    scalar_values = (
        config.hopping,
        config.g,
        config.basis_frequency,
        config.omega_p,
        config.K2,
        config.K4,
        config.temperature,
        config.bath_lambda,
        config.bath_gamma,
        config.dt_fit,
        config.t_fit_max,
        config.validation_dt,
        config.validation_t_max,
        config.quadrature_atol,
        config.quadrature_rtol,
        config.fit_sval_rtol,
        config.pole_stability_tolerance,
        config.duplicate_pole_tolerance,
        config.fit_absolute_tolerance,
        config.fit_relative_tolerance,
        config.final_time,
        config.time_step,
        config.tamen_tolerance,
        config.operator_tolerance,
        config.state_rounding_tolerance,
        config.trace_tolerance,
        config.hermiticity_tolerance,
        config.electron_number_tolerance,
    )
    all(isfinite, scalar_values) ||
        throw(ArgumentError("configuration values must be finite"))

    config.hopping >= 0 || throw(ArgumentError("hopping must be nonnegative"))
    config.basis_frequency > 0 ||
        throw(ArgumentError("basis_frequency must be positive"))
    config.d_raw > 0 || throw(ArgumentError("d_raw must be positive"))
    config.d_keep > 0 || throw(ArgumentError("d_keep must be positive"))
    config.d_raw >= config.d_keep ||
        throw(ArgumentError("d_raw must be at least d_keep"))
    config.omega_p > 0 || throw(ArgumentError("omega_p must be positive"))
    config.K4 >= 0 || throw(ArgumentError("K4 must be nonnegative"))
    config.temperature > 0 ||
        throw(ArgumentError("temperature must be positive"))
    config.bath_lambda >= 0 ||
        throw(ArgumentError("bath_lambda must be nonnegative"))
    config.bath_gamma > 0 ||
        throw(ArgumentError("bath_gamma must be positive"))
    config.dt_fit > 0 || throw(ArgumentError("dt_fit must be positive"))
    config.t_fit_max > 0 || throw(ArgumentError("t_fit_max must be positive"))
    config.validation_dt > 0 ||
        throw(ArgumentError("validation_dt must be positive"))
    config.validation_t_max > 0 ||
        throw(ArgumentError("validation_t_max must be positive"))
    config.quadrature_atol > 0 ||
        throw(ArgumentError("quadrature_atol must be positive"))
    config.quadrature_rtol > 0 ||
        throw(ArgumentError("quadrature_rtol must be positive"))
    config.quadrature_maxevals > 0 ||
        throw(ArgumentError("quadrature_maxevals must be positive"))
    config.fit_rank >= 0 || throw(ArgumentError("fit_rank must be nonnegative"))
    config.fit_rank_max > 0 ||
        throw(ArgumentError("fit_rank_max must be positive"))
    config.fit_rank == 0 || config.fit_rank <= config.fit_rank_max ||
        throw(ArgumentError("fit_rank must not exceed fit_rank_max"))
    config.fit_sval_rtol > 0 ||
        throw(ArgumentError("fit_sval_rtol must be positive"))
    config.pole_stability_tolerance >= 0 ||
        throw(ArgumentError("pole_stability_tolerance must be nonnegative"))
    config.duplicate_pole_tolerance >= 0 ||
        throw(ArgumentError("duplicate_pole_tolerance must be nonnegative"))
    config.fit_absolute_tolerance > 0 ||
        throw(ArgumentError("fit_absolute_tolerance must be positive"))
    config.fit_relative_tolerance > 0 ||
        throw(ArgumentError("fit_relative_tolerance must be positive"))
    config.hierarchy_nmax >= 0 ||
        throw(ArgumentError("hierarchy_nmax must be nonnegative"))
    config.final_time > 0 || throw(ArgumentError("final_time must be positive"))
    config.time_step > 0 || throw(ArgumentError("time_step must be positive"))
    config.temporal_basis_size >= 3 && isodd(config.temporal_basis_size) ||
        throw(ArgumentError("temporal_basis_size must be odd and at least three"))
    config.tamen_tolerance > 0 ||
        throw(ArgumentError("tamen_tolerance must be positive"))
    config.operator_tolerance > 0 ||
        throw(ArgumentError("operator_tolerance must be positive"))
    config.state_rounding_tolerance > 0 ||
        throw(ArgumentError("state_rounding_tolerance must be positive"))
    config.sweep_count > 0 || throw(ArgumentError("sweep_count must be positive"))
    config.local_iterations > 0 ||
        throw(ArgumentError("local_iterations must be positive"))
    config.kick_rank > 0 || throw(ArgumentError("kick_rank must be positive"))
    config.tt_max_rank > 0 || throw(ArgumentError("tt_max_rank must be positive"))
    config.trace_tolerance > 0 ||
        throw(ArgumentError("trace_tolerance must be positive"))
    config.hermiticity_tolerance > 0 ||
        throw(ArgumentError("hermiticity_tolerance must be positive"))
    config.electron_number_tolerance > 0 ||
        throw(ArgumentError("electron_number_tolerance must be positive"))
    config.progress_interval > 0 ||
        throw(ArgumentError("progress_interval must be positive"))

    step_count = config.final_time / config.time_step
    isapprox(step_count, round(step_count); atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("final_time must be an integer multiple of time_step"))
    return config
end
