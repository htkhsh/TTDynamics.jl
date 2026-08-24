using LinearAlgebra, Statistics
using TTSolver, TTDynamics, QFiND, HEOMKit, CairoMakie
using HEOMKit: icm2ifs

isdefined(@__MODULE__, :HolsteinConfig) || include("config.jl")
isdefined(@__MODULE__, :build_holstein_model) || include("model.jl")
isdefined(@__MODULE__, :run_holstein_dynamics) || include("dynamics.jl")
isdefined(@__MODULE__, :save_holstein_results) || include("plotting.jl")

# IMPORTANT: Before using these numerical defaults for production, converge the
# Padé order, TPSD tolerance, hierarchy local size, time step, and TT tolerances.
const DEFAULT_CONFIG = HolsteinConfig()

"""
    main(config=DEFAULT_CONFIG; output_directory=@__DIR__)

Decompose the bath, build and propagate the periodic Holstein HEOM-TT model,
and write its CSV and plot diagnostics beside this script.
"""
function main(config::HolsteinConfig=DEFAULT_CONFIG; output_directory=@__DIR__)
    validate_holstein_config(config)
    pole_decay_cm = config.brownian_damping_cm / 2
    println("Periodic Holstein model with independent Brownian baths")
    println(
        "  Sites=$(config.site_count), J=$(config.hopping_cm) cm⁻¹, " *
        "Ω=$(config.brownian_frequency_cm) cm⁻¹, " *
        "QFiND Γ_Q=$(config.brownian_damping_cm) cm⁻¹ " *
        "(pole decay Γ_Q/2=$pole_decay_cm cm⁻¹), " *
        "λ=$(config.reorganization_energy_cm) cm⁻¹, T=$(config.temperature_K) K",
    )
    println(
        "  Evolution: 0:$(config.time_step_fs):$(config.final_time_fs) fs; " *
        "hierarchy local size=$(config.hierarchy_local_size), " *
        "tAMEn tolerance=$(config.tamen_tolerance)",
    )
    println(
        "  TPSD: pade order=$(config.pade_order), " *
        "tolerance=$(config.tpsd_tolerance), type=$(config.pade_type)",
    )

    decomposition = decompose_holstein_bath(config)
    println(
        "  TPSD: $(length(decomposition.exponents)) terms, " *
        "validation relative error=" *
        "$(round(decomposition.relative_error, sigdigits=5))",
    )
    problem = build_holstein_model(config, decomposition)
    result = run_holstein_dynamics(config, problem)
    paths = save_holstein_results(output_directory, config, result)
    trace_drift = maximum(abs.(result.trace .- 1.0))
    println("  Maximum absolute trace drift: $(round(trace_drift, sigdigits=5))")
    println("  Wrote: $(paths.csv_path)")
    for path in paths.plot_paths
        println("  Wrote: $path")
    end
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
