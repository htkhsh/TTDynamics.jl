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

#### `HEOMParams`
Parameters for HEOM-TT calculation.

**Fields:**
- `nsys::Int`: Number of system states
- `Nh::Int`: Maximum hierarchy depth per bath mode
- `γ::Vector{ComplexF64}`: Exponential decay rates
- `c1::Vector{ComplexF64}`: BCF coefficients √cₖ
- `c2::Vector{ComplexF64}`: BCF coefficients √cₖ*
- `D::Vector{Matrix{ComplexF64}}`: Coupling operators
- `H_sys::Matrix{ComplexF64}`: System Hamiltonian

**Constructor:** `HEOMParams(nsys, Nh, γ, c, D, H_sys)`

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

#### `build_heom_liouvillian`
```julia
build_heom_liouvillian(params::HEOMParams; tol=1e-10) → (XOP, Ix, Pop)
```
Build HEOM Liouvillian super-operator in TT format.

HEOM equations:
```
∂ρₙ/∂t = -i Lsys ρₙ - Σₖ nₖ γₖ ρₙ 
         - i Σₖ [S, ρₙ₊ₑₖ] √((nₖ+1)|cₖ|)
         - i Σₖ (cₖ S ρₙ₋ₑₖ - cₖ* ρₙ₋ₑₖ S) √(nₖ/|cₖ|)
```

TT structure: `ρ_vec | b₁ | b₂ | ... | bₙ`

Returns:
- `XOP`: HEOM Liouvillian (TTMatrix)
- `Ix`: Trace operator (TTTensor)
- `Pop`: Population operators (Vector{TTTensor})

#### `build_initial_state`
```julia
build_initial_state(params::HEOMParams, init_state::Int; tol=1e-12) → TTTensor
```
Build initial density matrix |init_state⟩⟨init_state| ⊗ |0,0,...,0⟩.

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
payload. With `overwrite=true`, replacement follows filesystem rename semantics
and never recursively removes a directory destination.

`load_tt_binary` automatically detects whether a file contains a tensor or a
matrix. It rejects unknown format versions, malformed files, and files with
trailing content.

Version 1 stores fields in this order: the 8-byte magic `TTDYNBIN`, little-endian
`UInt16(1)` version, object `UInt8`, scalar `UInt8`, little-endian `UInt32` core
count, then each core's little-endian `UInt64` dimensions followed by its
column-major values. Complex values interleave real and imaginary components.
The format does not use Julia `Serialization`.

---

## Reference

HEOM-TT: Borrelli, Gelin, Chem. Phys. 481 (2016) 91-98
