module TTDynamics

using LinearAlgebra
using TTSolver
using HEOMKit

export PhysicalQuantity,
       Operator,
       Hamiltonian,
       SpinSpace,
       BosonSpace,
       PauliMatrices,
       PAULI,
       dim,
       # States
       localized_tt_state,
       initial_tt_state,  # alias for backward compatibility
       zero_tt_state,
       product_tt_state,
       lcbs_tt_state,
       tt_partial_trace,
       system_populations,
       # Density matrix states
       localized_dm_state,
       dm_state_from_matrix,
       product_dm_tt_state,
       localized_dm_tt_state,
       dm_tt_state,
       tt_svd,
       dm_to_tt,
       # Utility functions
       franck_condon_factors,
       estimate_basis_size,
       estimate_basis_sizes,
       save_tt_binary,
       load_tt_binary,
       # TFD types and functions
       BosonicTFD,
       BosonicEnv,
       SBSystem,
       tt_sbham,
       # HEOM-TT
       HEOMTTSystem,
       CorrelationFitMetadata,
       ExponentialCorrelation,
       heom_tt_dimensions,
       liouville_left,
       liouville_right,
       root_density_matrix,
       build_heom_liouvillian,
       build_initial_state,
       MultiCoreBathCoupling,
       MultiCoreHEOMTTSystem,
       multicore_physical_dimensions,
       multicore_heom_dimensions,
       build_multicore_liouvillian,
       build_multicore_heom_generator,
       build_multicore_heom_initial_state,
       root_multicore_state,
       build_multicore_observable,
       build_multicore_trace_observable,
       build_multicore_local_observable,
       multicore_expectation,
       MultiCoreRedfieldTTSystem,
       build_multicore_redfield_dissipator,
       build_multicore_redfield_generator

include("types.jl")
include("tt_states.jl")
include("util.jl")
include("tfd.jl")
include("heom.jl")
include("correlation.jl")
include("multicore_heom.jl")
include("multicore_states.jl")
include("multicore_redfield.jl")
include("tt_io.jl")

end
