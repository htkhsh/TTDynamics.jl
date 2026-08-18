# API Reference

## Types

### types.jl

#### `PhysicalQuantity`
Abstract type for physical quantities.

#### `Operator{T<:Complex}`
Wrapper for operator matrices with arithmetic operations.

**Fields:**
- `matrix::Matrix{T}`: The operator matrix

**Arithmetic:** `+`, `-`, `*`, `/`, `^`, `adjoint`

#### `Hamiltonian{T<:Complex}`
Wrapper for Hamiltonian matrices with arithmetic operations.

**Fields:**
- `matrix::Matrix{T}`: The Hamiltonian matrix

#### `SpinSpace`
Spin space with angular momentum operators.

**Fields:**
- `S::Float64`: Spin quantum number
- `Sx, Sy, Sz`: Spin operators
- `Sp, Sm`: Raising/lowering operators

**Constructor:** `SpinSpace(S::Real)`

#### `BosonSpace`
Bosonic Fock space with ladder operators.

**Fields:**
- `dim::Int`: Truncation dimension
- `a, adag`: Annihilation/creation operators
- `n`: Number operator
- `x, p`: Position/momentum operators (x = (a+a†)/√2)

**Constructor:** `BosonSpace(dim::Int)`

#### `PauliMatrices`
Pauli matrices container.

**Fields:** `σx`, `σy`, `σz`, `σp`, `σm`, `I`

**Global constant:** `PAULI`

### tfd.jl

#### `BosonicTFD`
Parameters for bosonic Thermo-Field Dynamics representation.

**Fields:**
- `nmode::Int`: Number of modes
- `freq::Vector{Float64}`: Mode frequencies
- `coeff::Vector{Float64}`: Coupling coefficients
- `V::Matrix{ComplexF64}`: System-bath coupling operator

#### `BosonicEnv`
Unified boson parameters from multiple environments.

**Fields:**
- `nenv::Int`: Number of environments
- `envs::Vector{BosonicTFD}`: List of baths
- `freq::Vector{Float64}`: All frequencies
- `coeff::Vector{Float64}`: All coefficients
- `nmode::Int`: Total number of modes
- `V::Vector{Matrix{ComplexF64}}`: Coupling operators

#### `SBSystem`
Spin-boson system.

**Fields:**
- `hsys::Matrix{ComplexF64}`: System Hamiltonian
- `benv::BosonicEnv`: Bosonic environment

**Constructors:**
- `SBSystem(hsys, benv)`
- `SBSystem(hsys, bfreq, bcoeff, bV)` - single bath
- `SBSystem(hsys, bfreqs, bcoeffs, bVs)` - multiple baths

### heom.jl

#### `HEOMTTSystem`
Parameters for a HEOM-TT calculation.

**Fields:**
- `H_sys::Matrix{ComplexF64}`: System Hamiltonian
- `noise::NoiseExp`: Exponential bath-correlation expansion and coupling operators
- `nb::Vector{Int}`: Local hierarchy dimensions

**Constructors:**
- `HEOMTTSystem(H_sys, noise, nb::Vector{Int})`
- `HEOMTTSystem(H_sys, noise, nb::Int)`

---

## Functions

### TT States (tt_states.jl)

#### `initial_tt_state`
```julia
initial_tt_state(dims::Vector{Int}; state_index=1, site=1) → TTTensor
initial_tt_state(spaces::Vector; state_index=1, site=1) → TTTensor
```
Create initial TT state with basis state at specified site.

- `dims`: Dimensions of each site
- `spaces`: Vector of SpinSpace/BosonSpace
- `state_index`: Basis state (1-indexed)
- `site`: Position in TT chain

#### `zero_tt_state`
```julia
zero_tt_state(dims::Vector{Int}; rk=1) → TTTensor
```
Create zero TT state with specified dimensions and ranks.

#### `product_tt_state`
```julia
product_tt_state(states::Vector{Vector{ComplexF64}}) → TTTensor
product_tt_state(states::Vector{<:AbstractVector}) → TTTensor
```
Create product state from local state vectors.

#### `lcbs_tt_state`
```julia
lcbs_tt_state(dims, coeffs; site=1, tol=1e-12) → TTTensor
```
Linear Combination of Basis States in TT format.

- `dims`: Site dimensions
- `coeffs`: Coefficients for each basis state
- `site`: Site for the superposition
- `tol`: Rounding tolerance

### TT Operators (tt_op.jl)

#### `local_tt_operator`
```julia
local_tt_operator(op, spaces, site) → TTMatrix
```
Build local operator: I₁ ⊗ ... ⊗ op ⊗ ... ⊗ Iₙ

- `op`: Local operator matrix
- `spaces`: Site dimensions (Int vector or Space vector)
- `site`: Position (1-indexed)

#### `identity_tt`
```julia
identity_tt(spaces) → TTMatrix
```
Build identity operator in TT format.

#### `sum_local_operators`
```julia
sum_local_operators(ops, spaces) → TTMatrix
```
Build Σᵢ Oᵢ where each Oᵢ acts on site i.

### TFD Hamiltonian (tfd.jl)

#### `build_sbham`
```julia
build_sbham(sys::SBSystem; basis_sizes=nothing, threshold=0.999, 
            nb_min=10, nb_max=30, unit_conv=1.0) → (TTMatrix, Vector{Int})
```
Build spin-boson Hamiltonian in TT format.

H = H_sys ⊗ I_bath + I_sys ⊗ Σᵢ ωᵢ nᵢ + Σⱼ Vⱼ ⊗ Σᵢ gᵢ xᵢ

- `sys`: SBSystem with hsys and benv
- `basis_sizes`: Fock basis sizes (auto-estimated if nothing)
- `threshold`: FC factor threshold for basis estimation
- `nb_min`, `nb_max`: Basis size bounds
- `unit_conv`: Unit conversion factor

Returns `(H_tt, basis_sizes)`.

### HEOM (heom.jl)

HEOM-TT states use the twin-space mode order
`[ket(N), bra(N), hierarchy...]`, rather than one vectorized `N^2` system
mode. A system left action `A ρ` acts on the ket mode, while a right action
`ρ B` acts on the bra mode through `transpose(B)`. The trace observable has
the same mode order and an internal ket-bra rank `N` that contracts matching
ket and bra indices.

The builder signatures are unchanged, but every HEOM state, observable, and
operator now has two system dimensions: `heom_tt_dimensions(system)` returns
`[N, N, nb...]`. Consequently, HEOM-TT checkpoints saved with the old
`[N^2, nb...]` layout are incompatible and must be regenerated, not relabeled
or automatically converted. Applications that save an HEOM-TT checkpoint must
store and validate the following application metadata alongside the TT data:

```text
heom_representation = "twin-space-v1"
```

The generic `TTTensor`/`TTMatrix` binary format itself is unchanged and does
not embed this HEOM-specific metadata. Generic non-HEOM TT binary files remain
compatible; the representation check belongs to the application that assigns
HEOM meaning to a loaded tensor.

#### `heom_tt_dimensions`
```julia
heom_tt_dimensions(system::HEOMTTSystem) -> Vector{Int}
```
Return the canonical HEOM-TT dimensions `[N, N, hierarchy...]`.

#### `root_density_matrix`
```julia
root_density_matrix(state::TTTensor, system::HEOMTTSystem) -> Matrix{ComplexF64}
```
Extract the root auxiliary density matrix by fixing all hierarchy modes to
their vacuum index. The state must have the twin-space dimensions returned by
`heom_tt_dimensions`; old vectorized HEOM states are rejected with a migration
error.

#### `build_heom_liouvillian`
```julia
build_heom_liouvillian(system::HEOMTTSystem; tol=1e-12) → (XOP, Ix, Pop)
```
Build HEOM Liouvillian super-operator in TT format.

HEOM equations:
```
∂ρₙ/∂t = -i Lsys ρₙ - Σₖ nₖ γₖ ρₙ 
         - i Σₖ [S, ρₙ₊ₑₖ] √((nₖ+1)|cₖ|)
         - i Σₖ (cₖ S ρₙ₋ₑₖ - cₖ* ρₙ₋ₑₖ S) √(nₖ/|cₖ|)
```

TT structure: `ket | bra | b₁ | b₂ | ... | bₙ`

Returns:
- `XOP`: HEOM Liouvillian (TTMatrix)
- `Ix`: Trace operator (TTTensor)
- `Pop`: Population operators (Vector{TTTensor})

#### `build_initial_state`
```julia
build_initial_state(system::HEOMTTSystem, init_state::Int; tol=1e-12) → TTTensor
```
Build the twin-space initial density matrix
`|init_state⟩_ket ⊗ |init_state⟩_bra ⊗ |0,0,...,0⟩`.

### Utilities (util.jl)

#### `franck_condon_factors`
```julia
franck_condon_factors(n, m, d) → Matrix{Float64}
```
Compute FC overlaps ⟨n|m'⟩ for displaced harmonic oscillator.

- `n`: Number of initial states
- `m`: Number of final (displaced) states
- `d`: Dimensionless displacement (g/ω)

Returns n×m matrix where F[i,j] = ⟨i-1|j-1'⟩.

#### `estimate_basis_size`
```julia
estimate_basis_size(freq, coupling; threshold=0.999, nb_min=10, nb_max=30) → Int
```
Estimate optimal Fock basis size based on FC factors.

#### `estimate_basis_sizes`
```julia
estimate_basis_sizes(freqs, couplings; threshold=0.999, nb_min=10, nb_max=30) → Vector{Int}
```
Estimate basis sizes for multiple modes.

### TT Binary I/O (tt_io.jl)

```julia
save_tt_binary(path, tt; overwrite=false) -> String
load_tt_binary(path) -> Union{TTTensor, TTMatrix}
```

`save_tt_binary` writes a tensor-train tensor or matrix in the versioned
TTDynamics binary format. It supports `Float32`, `Float64`, `ComplexF32`, and
`ComplexF64` core values. Existing targets are refused by default; set
`overwrite=true` to replace an existing file. By default, saving atomically
publishes only a completed sibling temporary file when the destination does not
exist, so a newly created destination is never left with a partially written
payload. With `overwrite=true`, a low-level Julia-runtime rename atomically
replaces a regular-file destination and never recursively removes a directory
destination.

`load_tt_binary` automatically detects whether a file contains a tensor or a
matrix. It rejects unknown format versions, malformed files, and files with
trailing content.

Version 1 stores fields in this order: the 8-byte magic `TTDYNBIN`, little-endian
`UInt16(1)` version, object-kind `UInt8`, scalar-type `UInt8`, little-endian
`UInt32` core count, then the core records. These published numeric codes are
permanent and must never be reassigned:

| Field | Code | Meaning |
| --- | ---: | --- |
| Object kind | `1` | `TTTensor` |
| Object kind | `2` | `TTMatrix` |
| Scalar type | `1` | `Float32` |
| Scalar type | `2` | `Float64` |
| Scalar type | `3` | `ComplexF32` |
| Scalar type | `4` | `ComplexF64` |

Each `TTTensor` core record has exactly three little-endian `UInt64`
dimensions in `(r_left, n, r_right)` order. Each `TTMatrix` core record has
exactly four in `(r_left, n, m, r_right)` order. The dimensions are followed by
column-major scalar values. Complex values interleave their real and imaginary
components. The format does not use Julia `Serialization`.

---

## Reference

HEOM-TT: Borrelli, Gelin, Chem. Phys. 481 (2016) 91-98
