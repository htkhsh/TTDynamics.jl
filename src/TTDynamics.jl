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
       heom_tt_dimensions,
       root_density_matrix,
       build_heom_liouvillian,
       build_initial_state

include("types.jl")
include("tt_states.jl")
include("util.jl")
include("tfd.jl")
include("heom.jl")
include("tt_io.jl")

end
