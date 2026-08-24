isdefined(@__MODULE__, :LatticeFrohlichConfig) || include("config.jl")
isdefined(@__MODULE__, :build_lattice_frohlich_model) || include("model.jl")
isdefined(@__MODULE__, :run_lattice_frohlich_dynamics) || include("dynamics.jl")
isdefined(@__MODULE__, :save_lattice_frohlich_results) || include("plotting.jl")

const DEFAULT_LATTICE_FROHLICH_CONFIG = LatticeFrohlichConfig()

function lattice_frohlich_main(config::LatticeFrohlichConfig=
                               DEFAULT_LATTICE_FROHLICH_CONFIG;
                               output_directory=@__DIR__)
    decomposition = decompose_lattice_frohlich_bath(config)
    problem = build_lattice_frohlich_model(config, decomposition)
    result = run_lattice_frohlich_dynamics(config, problem)
    save_lattice_frohlich_results(output_directory, config, result)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    lattice_frohlich_main()
end
