using LinearAlgebra
using TTDynamics
using TTSolver

function _quartic_adjoint_root(root::TTTensor, physical_dimensions::Vector{Int})
    length(root.cores) == length(physical_dimensions) || throw(ArgumentError(
        "root ADO core count must match the physical dimensions",
    ))
    cores = Vector{Array{ComplexF64,3}}(undef, length(root.cores))
    for (index, (core, liouville_dimension)) in enumerate(zip(root.cores, physical_dimensions))
        hilbert_dimension = isqrt(liouville_dimension)
        hilbert_dimension^2 == liouville_dimension || throw(ArgumentError(
            "physical dimension $liouville_dimension is not a square Liouville dimension",
        ))
        adjoint_core = similar(ComplexF64.(core))
        for column in 1:hilbert_dimension, row in 1:hilbert_dimension
            destination = row + (column - 1) * hilbert_dimension
            source = column + (row - 1) * hilbert_dimension
            adjoint_core[:, destination, :] .= conj.(core[:, source, :])
        end
        cores[index] = adjoint_core
    end
    return TTTensor(cores)
end

function _quartic_finite_complex(value)
    return isfinite(real(value)) && isfinite(imag(value))
end

"""
    measure_quartic_state(state, model)

Measure physical observables on the hierarchy root ADO and return TT-rank and
Hermiticity diagnostics without expanding the hierarchy.
"""
function measure_quartic_state(state::TTTensor, model::QuarticModel)
    root = root_ado(state, model.system)
    observables = model.observables
    trace_value = root_expectation(state, model.system, observables.trace)
    populations = real.([
        root_expectation(state, model.system, observable)
        for observable in observables.electron_populations
    ])
    electron_number = root_expectation(
        state,
        model.system,
        observables.electron_number,
    )
    oscillator_q = real.([
        root_expectation(state, model.system, observable)
        for observable in observables.oscillator_q
    ])
    oscillator_q2 = real.([
        root_expectation(state, model.system, observable)
        for observable in observables.oscillator_q2
    ])
    oscillator_energy = real.([
        root_expectation(state, model.system, observable)
        for observable in observables.oscillator_hamiltonian
    ])
    adjoint_root = _quartic_adjoint_root(root, model.system.physical_dimensions)
    hermiticity_error = Float64(real(tt_norm(root - adjoint_root)))
    ranks = collect(tt_ranks(state))
    mean_rank = sum(ranks) / length(ranks)

    all(isfinite, populations) || throw(ArgumentError("electron populations must be finite"))
    _quartic_finite_complex(trace_value) || throw(ArgumentError("root trace must be finite"))
    _quartic_finite_complex(electron_number) || throw(ArgumentError(
        "electron number must be finite",
    ))
    all(isfinite, oscillator_q) || throw(ArgumentError("oscillator q values must be finite"))
    all(isfinite, oscillator_q2) || throw(ArgumentError("oscillator q2 values must be finite"))
    all(isfinite, oscillator_energy) || throw(ArgumentError(
        "oscillator energy values must be finite",
    ))
    isfinite(hermiticity_error) || throw(ArgumentError("Hermiticity error must be finite"))

    return (;
        populations,
        trace=ComplexF64(trace_value),
        electron_number=ComplexF64(electron_number),
        oscillator_q,
        oscillator_q2,
        oscillator_energy,
        hermiticity_error,
        maximum_rank=maximum(ranks),
        mean_rank=Float64(mean_rank),
    )
end

function _quartic_empty_result(state::TTTensor, measurement)
    return (;
        times=Float64[0.0],
        populations=reshape(copy(measurement.populations), :, 1),
        trace=ComplexF64[measurement.trace],
        electron_number=ComplexF64[measurement.electron_number],
        oscillator_q=reshape(copy(measurement.oscillator_q), :, 1),
        oscillator_q2=reshape(copy(measurement.oscillator_q2), :, 1),
        oscillator_energy=reshape(copy(measurement.oscillator_energy), :, 1),
        hermiticity_error=Float64[measurement.hermiticity_error],
        maximum_rank=Int[measurement.maximum_rank],
        mean_rank=Float64[measurement.mean_rank],
        tamen_residual=Union{Missing,Float64}[missing],
        tamen_truncation=Union{Missing,Float64}[missing],
        final_state=state,
    )
end

function _quartic_root_trace_observable(model::QuarticModel)
    cores = [ComplexF64.(copy(core)) for core in model.observables.trace.cores]
    for dimension in model.system.hierarchy_sizes
        vacuum = zeros(ComplexF64, dimension)
        vacuum[1] = 1
        push!(cores, reshape(vacuum, 1, dimension, 1))
    end
    return TTTensor(cores)
end

function _quartic_validate_step_grid(final_time::Real, time_step::Real)
    isfinite(final_time) && final_time >= 0 || throw(ArgumentError(
        "final_time must be finite and nonnegative",
    ))
    isfinite(time_step) && time_step > 0 || throw(ArgumentError(
        "time_step must be finite and positive",
    ))
    ratio = final_time / time_step
    step_count = round(Int, ratio)
    isapprox(ratio, step_count; atol=1e-12, rtol=1e-12) || throw(ArgumentError(
        "final_time must be an integer multiple of time_step",
    ))
    return step_count
end

function _quartic_warn_diagnostics(measurement, config::QuarticConfig, time::Float64)
    abs(measurement.trace - 1) > config.trace_tolerance && @warn(
        "quartic root trace exceeded tolerance",
        time,
        trace=measurement.trace,
        tolerance=config.trace_tolerance,
    )
    measurement.hermiticity_error > config.hermiticity_tolerance && @warn(
        "quartic root Hermiticity error exceeded tolerance",
        time,
        error=measurement.hermiticity_error,
        tolerance=config.hermiticity_tolerance,
    )
    abs(measurement.electron_number - 1) > config.electron_number_tolerance && @warn(
        "quartic electron number exceeded tolerance",
        time,
        electron_number=measurement.electron_number,
        tolerance=config.electron_number_tolerance,
    )
    return nothing
end

function _quartic_collect_result(times, measurements, final_state::TTTensor)
    return (;
        times,
        populations=hcat(getfield.(measurements, :populations)...),
        trace=ComplexF64.(getfield.(measurements, :trace)),
        electron_number=ComplexF64.(getfield.(measurements, :electron_number)),
        oscillator_q=hcat(getfield.(measurements, :oscillator_q)...),
        oscillator_q2=hcat(getfield.(measurements, :oscillator_q2)...),
        oscillator_energy=hcat(getfield.(measurements, :oscillator_energy)...),
        hermiticity_error=Float64.(getfield.(measurements, :hermiticity_error)),
        maximum_rank=Int.(getfield.(measurements, :maximum_rank)),
        mean_rank=Float64.(getfield.(measurements, :mean_rank)),
        tamen_residual=Union{Missing,Float64}[missing for _ in times],
        tamen_truncation=Union{Missing,Float64}[missing for _ in times],
        final_state,
    )
end

"""
    propagate_quartic_heom(model, config=model.config; final_time=config.final_time)

Propagate a quartic HEOM-TT model. A zero final time returns only the initial
root measurement.
"""
function propagate_quartic_heom(
    model::QuarticModel,
    config::QuarticConfig=model.config;
    final_time::Real=config.final_time,
)
    step_count = _quartic_validate_step_grid(final_time, config.time_step)
    iszero(step_count) && return _quartic_empty_result(
        model.initial_state,
        measure_quartic_state(model.initial_state, model),
    )

    options = Dict{Symbol,Any}(
        :verb => 0,
        :nswp => config.sweep_count,
        :local_iters => config.local_iterations,
        :kickrank => config.kick_rank,
        :time_scheme => "CN",
        :time_error_damp => 100.0,
        :obs => [_quartic_root_trace_observable(model)],
    )
    times = collect(range(0.0; step=config.time_step, length=step_count + 1))
    state = model.initial_state
    measurements = [measure_quartic_state(state, model)]
    _quartic_warn_diagnostics(first(measurements), config, 0.0)
    rank_warning_emitted = false
    if first(measurements).maximum_rank >= config.tt_max_rank
        @warn(
            "quartic TT rank reached the configured warning threshold",
            time=0.0,
            rank=first(measurements).maximum_rank,
            threshold=config.tt_max_rank,
        )
        rank_warning_emitted = true
    end

    for step in 1:step_count
        space_time_state = tkron(
            state,
            tt_ones(config.temporal_basis_size; T=ComplexF64),
        )
        space_time_state, time_grid, _, _, _ = tamen(
            space_time_state,
            config.time_step * model.generator,
            config.tamen_tolerance,
            options,
        )
        state = extract_snapshot(space_time_state, time_grid, 1.0, "CN")
        state = tt_round(state, config.state_rounding_tolerance)
        all(core -> all(isfinite, core), state.cores) || throw(ArgumentError(
            "quartic HEOM-TT state must remain finite",
        ))

        measurement = measure_quartic_state(state, model)
        push!(measurements, measurement)
        _quartic_warn_diagnostics(measurement, config, times[step + 1])
        if !rank_warning_emitted && measurement.maximum_rank >= config.tt_max_rank
            @warn(
                "quartic TT rank reached the configured warning threshold",
                time=times[step + 1],
                rank=measurement.maximum_rank,
                threshold=config.tt_max_rank,
            )
            rank_warning_emitted = true
        end
    end
    return _quartic_collect_result(times, measurements, state)
end

"""
    quartic_output_paths(directory; stem="quartic")

Return deterministic CSV and plot paths for one quartic dynamics run.
"""
function quartic_output_paths(directory::AbstractString; stem::AbstractString="quartic")
    isempty(stem) && throw(ArgumentError("stem must be nonempty"))
    return (;
        csv=joinpath(directory, "$(stem).csv"),
        populations=joinpath(directory, "$(stem)_populations.png"),
        oscillator_q=joinpath(directory, "$(stem)_q.png"),
        oscillator_q2=joinpath(directory, "$(stem)_q2.png"),
        oscillator_energy=joinpath(directory, "$(stem)_energy.png"),
        trace=joinpath(directory, "$(stem)_trace.png"),
        rank=joinpath(directory, "$(stem)_rank.png"),
    )
end

function _quartic_atomic_rename(oldpath::AbstractString, newpath::AbstractString)
    old = String(oldpath)
    new = String(newpath)
    status = ccall(:jl_fs_rename, Int32, (Cstring, Cstring), old, new)
    status < 0 && Base.uv_error("rename($(repr(old)), $(repr(new)))", status)
    return nothing
end

function _quartic_csv_header(result)
    site_count = size(result.populations, 1)
    oscillator_count = size(result.oscillator_q, 1)
    return join([
        "time",
        ["population_site_$site" for site in 1:site_count]...,
        "trace_real",
        "trace_imag",
        "hermiticity_error",
        "electron_number_real",
        "electron_number_imag",
        ["oscillator_q_site_$site" for site in 1:oscillator_count]...,
        ["oscillator_q2_site_$site" for site in 1:oscillator_count]...,
        ["oscillator_energy_site_$site" for site in 1:oscillator_count]...,
        "maximum_rank",
        "mean_rank",
        "tamen_residual",
        "tamen_truncation",
    ], ",")
end

"""
    write_quartic_csv(path, result; overwrite=false)

Atomically write quartic populations and diagnostics. Existing targets are
rejected unless `overwrite=true` is explicitly requested.
"""
function write_quartic_csv(
    path::AbstractString,
    result;
    overwrite::Bool=false,
)::String
    target = String(path)
    !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
    temporary_path = nothing
    try
        temporary_path, io = mktemp(dirname(abspath(target)))
        try
            println(io, _quartic_csv_header(result))
            for time_index in eachindex(result.times)
                residual = result.tamen_residual[time_index]
                truncation = result.tamen_truncation[time_index]
                row = Any[
                    result.times[time_index],
                    result.populations[:, time_index]...,
                    real(result.trace[time_index]),
                    imag(result.trace[time_index]),
                    result.hermiticity_error[time_index],
                    real(result.electron_number[time_index]),
                    imag(result.electron_number[time_index]),
                    result.oscillator_q[:, time_index]...,
                    result.oscillator_q2[:, time_index]...,
                    result.oscillator_energy[:, time_index]...,
                    result.maximum_rank[time_index],
                    result.mean_rank[time_index],
                    ismissing(residual) ? "" : residual,
                    ismissing(truncation) ? "" : truncation,
                ]
                println(io, join(row, ","))
            end
        finally
            close(io)
        end
        !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
        if overwrite
            _quartic_atomic_rename(temporary_path, target)
            temporary_path = nothing
        else
            Base.Filesystem.hardlink(temporary_path, target)
        end
        return target
    finally
        temporary_path === nothing || rm(temporary_path; force=true)
    end
end
