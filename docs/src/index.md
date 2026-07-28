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
H_tt, basis_sizes = build_sbham(sys; unit_conv=icm2ifs)

# Initial state |↑⟩ ⊗ |0,0,...⟩
spaces = vcat([2], basis_sizes)
psi0 = initial_tt_state(spaces; state_index=1, site=1)

# Time evolution with tAMEn
A = -1im * H_tt
psi_t = tamen(A, psi0, ts; opts...)
```

## Contents

```@contents
Pages = ["api.md"]
```

```@index
```
