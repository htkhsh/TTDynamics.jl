using LinearAlgebra
using Statistics
using TTSolver
using TTDynamics
using QFiND
using HEOMKit
using HEOMKit: icm2ifs
using CairoMakie

if !isdefined(@__MODULE__, :HolsteinConfig)
    include("../holstein/utils.jl")
end
if !isdefined(@__MODULE__, :decompose_brownian_bcf)
    include("../holstein/holstein_brownian_heomtt.jl")
end
if !isdefined(@__MODULE__, :normalized_frohlich_kernel)
    include("utils.jl")
end

const DEFAULT_LATTICE_FROHLICH_CONFIG = HolsteinConfig()

function build_lattice_frohlich_heomtt(config::HolsteinConfig, decomposition;
                                        kernel=default_frohlich_kernel)
    validate_config(config)
    H_fs = periodic_holstein_hamiltonian(
        config.site_energies_cm,
        config.hopping_cm,
    ) * icm2ifs
    couplings = frohlich_coupling_operators(config.site_count; kernel)
    baths = [
        BathExp(decomposition.exponents, decomposition.coefficients, coupling)
        for coupling in couplings
    ]
    system = HEOMTTSystem(H_fs, NoiseExp(baths), config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable, population_observables)
end
