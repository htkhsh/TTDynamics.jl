```@meta
CurrentModule = TTDynamics
```

# TTDynamics

Tensor-Train methods for open quantum system dynamics.

## Features

- **TT States**: Initial states, product states, superposition states
- **TT Operators**: Local operators, identity, sums
- **TFD-TT**: Thermo-Field Dynamics in TT format
- **HEOM-TT**: Hierarchical Equations of Motion in TT format
- **Time-dependent propagation**: Separated Hamiltonians with KSL and tAMEn

## Quick Start

```julia
using TTDynamics
using TTSolver

# Spin-boson model
σz = Matrix(PAULI.σz)
σx = Matrix(PAULI.σx)
hsys = 0.5 * ε * σz + Δ * σx

# Create system
sys = SBSystem(hsys, ω, g, σz)

# Build Hamiltonian (basis sizes auto-estimated)
H_tt, basis_sizes = tt_sbham(sys; unit_conv=icm2ifs)

# Initial state |↑⟩ ⊗ |0,0,...⟩
spaces = vcat([2], basis_sizes)
psi0 = localized_tt_state(spaces)

# Time evolution with tAMEn
psi_t = propagate_tamen(H_tt, psi0, ts; options=Dict(:verb => 0))
```

## Contents

```@contents
Pages = ["api.md"]
```

```@index
```
