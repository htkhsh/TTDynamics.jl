using LinearAlgebra
using TTDynamics

isdefined(@__MODULE__, :QuarticConfig) || include(joinpath(@__DIR__, "config.jl"))
isdefined(@__MODULE__, :QuarticMode) || include(joinpath(@__DIR__, "oscillator.jl"))
isdefined(@__MODULE__, :CriticallyDampedBrownian) || include(joinpath(@__DIR__, "correlation.jl"))
isdefined(@__MODULE__, :QuarticModel) || include(joinpath(@__DIR__, "model.jl"))
isdefined(@__MODULE__, :propagate_quartic_heom) || include(joinpath(@__DIR__, "dynamics.jl"))
isdefined(@__MODULE__, :plot_one_site_result) || include(joinpath(@__DIR__, "plotting.jl"))

if !isdefined(@__MODULE__, :DEFAULT_QUARTIC_ONE_SITE_CONFIG)
    const DEFAULT_QUARTIC_ONE_SITE_CONFIG = QuarticConfig(
        site_count=1,
        site_energies=[0.0],
        hopping=0.0,
    )
end

function _quartic_workflow_config_with(config::QuarticConfig; kwargs...)
    names = fieldnames(QuarticConfig)
    values = NamedTuple{names}(Tuple(getfield(config, name) for name in names))
    return QuarticConfig(; merge(values, kwargs)...)
end

"""
    build_quartic_correlation_fit(config)

Sample the configured cBO correlation on a uniform training grid and a
disjoint holdout grid, then return the validated common-basis ESPRIT fit.
"""
function build_quartic_correlation_fit(config::QuarticConfig)
    validate_quartic_config(config)
    config.bath_lambda > 0 || throw(ArgumentError(
        "bath_lambda must be positive when constructing a bath fit",
    ))
    bath = CriticallyDampedBrownian(config.bath_lambda, config.bath_gamma)
    training_times = collect(0.0:config.dt_fit:config.t_fit_max)
    validation_candidates = collect(
        config.validation_dt / 2:config.validation_dt:config.validation_t_max,
    )
    validation_times = filter(validation_candidates) do validation_time
        !any(
            training_time -> isapprox(
                validation_time,
                training_time;
                atol=1e-12,
                rtol=1e-12,
            ),
            training_times,
        )
    end
    isempty(validation_times) && throw(ArgumentError(
        "validation grid must contain times disjoint from the training grid",
    ))
    sampling_keywords = (
        rtol=config.quadrature_rtol,
        atol=config.quadrature_atol,
        maxevals=config.quadrature_maxevals,
    )
    training_samples = sample_bath_correlation(
        bath,
        config.temperature,
        training_times;
        sampling_keywords...,
    )
    validation_samples = sample_bath_correlation(
        bath,
        config.temperature,
        validation_times;
        sampling_keywords...,
    )
    return fit_correlation_esprit(
        training_samples,
        training_times;
        fit_rank=config.fit_rank,
        fit_rank_max=config.fit_rank_max,
        fit_sval_rtol=config.fit_sval_rtol,
        pole_stability_tolerance=config.pole_stability_tolerance,
        duplicate_pole_tolerance=config.duplicate_pole_tolerance,
        fit_absolute_tolerance=config.fit_absolute_tolerance,
        fit_relative_tolerance=config.fit_relative_tolerance,
        validation_samples,
        validation_times,
    )
end

function _print_quartic_fit(io::IO, fit::ExponentialCorrelation)
    metadata = fit.metadata
    println(io, "rates = $(repr(fit.rates))")
    println(io, "coeff_forward = $(repr(fit.coeff_forward))")
    println(io, "coeff_backward = $(repr(fit.coeff_backward))")
    println(io, "training_absolute_error = $(metadata.training_absolute_error)")
    println(io, "training_relative_error = $(metadata.training_relative_error)")
    println(io, "validation_absolute_error = $(metadata.validation_absolute_error)")
    println(io, "validation_relative_error = $(metadata.validation_relative_error)")
    println(io, "vandermonde_condition = $(metadata.vandermonde_condition)")
    println(io, "minimum_pole_separation = $(metadata.minimum_pole_separation)")
    return nothing
end

function _quartic_preflight_outputs(paths; overwrite::Bool)
    overwrite && return nothing
    for path in paths
        ispath(path) && throw(ArgumentError("target already exists: $path"))
    end
    return nothing
end

function _quartic_displaced_initial_model(model::QuarticModel, displacement::Real)
    shift = Float64(displacement)
    isfinite(shift) || throw(ArgumentError("displacement must be finite"))
    model.config.site_count == 1 || throw(ArgumentError(
        "a displaced initial oscillator state is supported only by the one-site workflow",
    ))
    ground = zeros(ComplexF64, model.config.d_keep)
    ground[1] = 1
    displaced = exp(-1im * shift * model.mode.p) * ground
    displaced ./= norm(displaced)
    oscillator_density = displaced * displaced'
    electron_density = ComplexF64[1;;]
    initial_state = build_multicore_heom_initial_state(
        model.system,
        [electron_density, oscillator_density];
        tol=model.config.state_rounding_tolerance,
    )
    return QuarticModel(
        model.config,
        model.mode,
        model.fit,
        model.system_hamiltonian,
        model.system,
        model.generator,
        initial_state,
        model.observables,
    )
end

"""
    one_site_main(config=DEFAULT_QUARTIC_ONE_SITE_CONFIG; kwargs...)

Run one local oscillator with one electronic state, save its CSV diagnostics,
and optionally render plots. Builders and the propagator are injectable so a
workflow smoke test need not perform quadrature or time evolution. Plotters
may accept a third positional publication callback for incremental cleanup;
legacy two-argument plotters remain supported.
"""
function one_site_main(
    config::QuarticConfig=DEFAULT_QUARTIC_ONE_SITE_CONFIG;
    output_directory::AbstractString=joinpath(@__DIR__, "output", "one_site"),
    overwrite::Bool=false,
    displacement::Union{Nothing,Real}=nothing,
    correlation_builder=build_quartic_correlation_fit,
    mode_builder=build_quartic_mode,
    model_builder=build_quartic_model,
    propagator=propagate_quartic_heom,
    csv_writer=write_quartic_csv,
    plotter=plot_one_site_result,
    progress_io::IO=stdout,
)
    validate_quartic_config(config)
    config.site_count == 1 || throw(ArgumentError("one_site_main requires site_count = 1"))
    paths = quartic_output_paths(String(output_directory); stem="one_site")
    plot_paths = (
        paths.populations,
        paths.oscillator_q,
        paths.oscillator_q2,
        paths.oscillator_energy,
        paths.trace,
        paths.rank,
    )
    expected_paths = plotter === nothing ? (paths.csv,) : (paths.csv, plot_paths...)
    _quartic_preflight_outputs(expected_paths; overwrite)

    fit = iszero(config.bath_lambda) ? nothing : correlation_builder(config)
    fit === nothing || _print_quartic_fit(progress_io, fit)
    mode = mode_builder(config)
    model = model_builder(config, mode, fit; initial_site=1)
    if displacement !== nothing
        model = _quartic_displaced_initial_model(model, displacement)
    end
    result = propagator(model, config)

    mkpath(String(output_directory))
    published_paths = String[]
    try
        csv_writer(paths.csv, result; overwrite)
        push!(published_paths, paths.csv)
        if plotter !== nothing
            publication_callback = function(path)
                published_path = String(path)
                published_path in plot_paths || throw(ArgumentError(
                    "plotter reported an unexpected published path: $published_path",
                ))
                push!(published_paths, published_path)
                return nothing
            end
            if applicable(plotter, paths, result, publication_callback)
                plotter(paths, result, publication_callback; overwrite)
            else
                plotter(paths, result; overwrite)
                append!(published_paths, plot_paths)
            end
        end
    catch
        if !overwrite
            for path in published_paths
                rm(path; force=true)
            end
        end
        rethrow()
    end
    println(progress_io, "Wrote: $(paths.csv)")
    if plotter !== nothing
        for path in plot_paths
            isfile(path) && println(progress_io, "Wrote: $path")
        end
    end
    return (; config, mode, fit, model, result, paths)
end

if abspath(PROGRAM_FILE) == @__FILE__
    one_site_main()
end
