# Multi-Subsystem Twin-Core HEOM-TT Design

## Goal

Extend TTDynamics with an HEOM-TT representation for a physical Hilbert space
factored across multiple TT cores while keeping every subsystem's ket and bra
indices in separate TT cores. This supplies the dynamics foundation needed by
ChargeTransportModels for a one-charge core coupled to multiple explicit local
vibrational cores.

The implementation preserves the existing `HEOMTTSystem`, whose two physical
cores represent the ket and bra of one dense system, and the existing
`MultiCoreHEOMTTSystem`, whose physical cores use fused local Liouville
dimensions. The new API is additive.

## Meaning of a Subsystem

A subsystem is one tensor factor of the physical Hilbert space,

```math
\mathcal H = \bigotimes_{j=1}^{M} \mathcal H_j,
```

and therefore one core of the Hilbert-space Hamiltonian MPO. A subsystem need
not be dynamically independent. Interactions between subsystems are carried by
the virtual bonds of the Hamiltonian MPO.

For the intended charge-transport use, the subsystem list is one charge core
followed by one core per explicit vibrational mode. HEOM hierarchy modes are
auxiliary modes and are not physical subsystems.

## Tensor Layout

If the Hilbert-space dimensions are `[d1, d2, ..., dM]`, the physical twin-space
layout is interleaved:

```text
[ket_1, bra_1, ket_2, bra_2, ..., ket_M, bra_M]
```

and has dimensions:

```text
[d1, d1, d2, d2, ..., dM, dM]
```

HEOM hierarchy cores follow all physical cores:

```text
[ket_1, bra_1, ..., ket_M, bra_M, hierarchy_1, ...]
```

Interleaving keeps the ket-bra correlation of each local mixed state on the
bond inside its subsystem pair. Placing all ket cores before all bra cores is
not supported by this API because product thermal states can then require a
rank across the central bond that grows as a product of local density-matrix
ranks.

## Public Types

### `TwinCoreBathCoupling`

```julia
struct TwinCoreBathCoupling
    subsystem::Int
    operator::Matrix{ComplexF64}
    correlation::ExponentialCorrelation
end
```

`subsystem` indexes the Hilbert-space MPO cores, not the interleaved twin-space
cores. `operator` is a finite Hermitian local system operator with dimensions
matching that subsystem. `correlation` reuses the existing scaled exponential
correlation representation.

The constructor validates a positive subsystem index, a nonempty square finite
Hermitian operator, and delegates correlation validation to
`ExponentialCorrelation`.

### `TwinCoreHEOMTTSystem`

```julia
struct TwinCoreHEOMTTSystem
    hamiltonian::TTMatrix{ComplexF64}
    hilbert_dimensions::Vector{Int}
    couplings::Vector{TwinCoreBathCoupling}
    hierarchy_sizes::Vector{Int}
end
```

The preferred constructor derives `hilbert_dimensions` from the square local
dimensions of `hamiltonian`:

```julia
TwinCoreHEOMTTSystem(hamiltonian, couplings, hierarchy_sizes)
```

It converts the MPO to `ComplexF64`, checks finite cores and square positive
local dimensions, validates every coupling against its target subsystem, and
requires one positive hierarchy size per exponential correlation term across
all couplings. A convenience constructor accepts one uniform hierarchy size.

The constructor does not expand the MPO to test global Hermiticity. Model
builders must construct a Hermitian Hamiltonian; small dense equivalence tests
cover the package's own construction paths.

## Hamiltonian Twin-Space Embedding

Let the Hamiltonian MPO core for subsystem `j` be

```text
G_j[alpha_(j-1), s_out, s_in, alpha_j].
```

The ket-side operator `H_left` places `G_j` on `ket_j`. The following `bra_j`
core applies the physical identity and propagates the outgoing MPO bond
`alpha_j` unchanged. The bra-side operator `H_right` first propagates the
incoming MPO bond through an identity on `ket_j`, then places the physical
transpose of `G_j` on `bra_j`.

This produces the interleaved equivalents of

```math
H^L \rho = H\rho,
\qquad
H^R \rho = \rho H,
```

without forming a local dimension `d_j^2`. The physical Liouvillian is

```math
\mathcal L_H = -i(H^L-H^R).
```

The public helper is:

```julia
build_twin_liouvillian(hamiltonian; tol=1e-12)
```

It returns a `TTMatrix` over the interleaved twin-space dimensions and rounds
only after adding the two exactly embedded MPOs. `tol` must be finite and
nonnegative.

## HEOM Generator

For coupling `a`, local operator `V_a`, exponential pole `k`, and hierarchy
ladder operators `n`, `bminus`, and `bplus`, the generator uses the same scaled
HEOM convention as `build_multicore_heom_generator`:

```math
-\gamma_{ak} n_{ak}
-i s_{ak}(V_a^L-V_a^R)b^-_{ak}
-\frac{i}{s_{ak}}
  (c^{\rightarrow}_{ak}V_a^L-c^{\leftarrow}_{ak}V_a^R)b^+_{ak}.
```

Here `V_a^L` acts on the target subsystem's ket core and `V_a^R` uses
`transpose(V_a)` on its bra core. Every other physical and hierarchy core is
an identity, except the selected hierarchy ladder core.

The public builder is:

```julia
build_twin_heom_generator(system; tol=1e-12)
```

It appends hierarchy identity cores to the physical twin Liouvillian and adds
decay, upward, and downward terms incrementally with TT rounding. Construction
never expands a global physical or hierarchy operator.

The public dimension helper is:

```julia
twin_core_heom_dimensions(system)
```

and returns the full physical-plus-hierarchy layout.

## Initial Density States

The primary constructor accepts one local density matrix per subsystem:

```julia
build_twin_heom_initial_state(
    system,
    local_density_matrices;
    tol=1e-12,
    density_tolerance=1e-10,
)
```

Each local density matrix is validated as finite, Hermitian, positive
semidefinite within `density_tolerance`, and unit trace within that tolerance.
An SVD factors it into adjacent ket and bra TT cores. Consequently the internal
ket-bra bond rank equals its numerical matrix rank, using
`density_tolerance * maximum(singular_values)` as the relative cutoff, rather
than forcing a local dimension squared. The subsystem pairs are concatenated,
and a vacuum vector is appended for every hierarchy core.

This constructor intentionally creates factorized physical initial states.
An overload accepts an already constructed physical twin-space `TTTensor` for
correlated initial states, validates its dimensions, and appends hierarchy
vacua.

## Root State and Observables

```julia
root_twin_state(state, system)
```

fixes every hierarchy index to its vacuum value and returns a `TTTensor` over
the interleaved physical twin cores. It does not construct a dense density
matrix.

```julia
build_twin_observable(local_operators; tol=1e-12)
```

builds the Hilbert-Schmidt dual TT vector for a product observable. It uses
the local matrix `adjoint(O_j)` so that `tt_dot(observable, root_state)` equals
`tr(O * rho)` under TTSolver's conjugating dot product. Identities may be
supplied explicitly, and convenience trace and single-subsystem observable
builders use the same path.

```julia
twin_expectation(observable, state, system)
```

contracts a compatible physical observable with the hierarchy root without
dense expansion.

For diagnostics and tests only,

```julia
root_density_matrix(state, system; max_dimension=2_000)
```

expands the root twin tensor, permutes interleaved indices into all-ket then
all-bra order, and reshapes it to a dense matrix. It rejects a total Hilbert
dimension above `max_dimension` unless the caller explicitly increases the
limit.

## API Placement

The implementation lives in a focused new source file, included and exported
from `src/TTDynamics.jl`. It reuses internal validation and local-product
helpers where their existing contracts fit, but does not change the behavior
or storage of the two existing HEOM-TT system types.

ChargeTransportModels will later construct a Hilbert-space Hamiltonian MPO and
`TwinCoreBathCoupling` objects through an optional TTDynamics extension. No
charge-transport model knowledge belongs in TTDynamics.

## Error Handling

Public constructors and builders throw `ArgumentError` or `DimensionMismatch`
before expensive TT construction for:

- a nonsquare Hamiltonian MPO or nonpositive local dimension;
- nonfinite Hamiltonian cores;
- a coupling subsystem outside `1:M`;
- a coupling operator with the wrong dimensions;
- a nonfinite or non-Hermitian coupling operator;
- hierarchy-size counts that do not match correlation poles;
- nonpositive hierarchy sizes;
- invalid tolerances;
- incompatible physical or full-state TT dimensions;
- invalid local density matrices;
- unsafe dense root-state expansion.

Error messages identify the subsystem, coupling, hierarchy mode, or TT core
that failed validation.

## Testing

Focused tests cover:

1. twin-space dimensions for heterogeneous subsystem sizes;
2. ket and bra MPO embeddings for rank-one and nontrivial-rank Hamiltonian
   MPOs;
3. dense equality of `build_twin_liouvillian` with `-i[H, rho]` after the
   documented index permutation;
4. one-subsystem generator equivalence with the existing `HEOMTTSystem` under
   identical exponential correlations and hierarchy sizes;
5. two-subsystem HEOM generator equality with a direct dense term
   construction;
6. pure and mixed product density states, including a full-rank thermal local
   state whose ket-bra bond rank is its matrix rank;
7. correlated physical twin-state input and hierarchy-vacuum appending;
8. trace, local expectation values, root extraction, and guarded dense root
   reconstruction;
9. all validation failures listed above;
10. unchanged existing HEOM, multi-core HEOM, quartic, Holstein, and
    Lattice-Frohlich tests.

The focused test first uses small dimensions that permit exact `tt_full`
comparisons. The full TTDynamics suite then checks backward compatibility.

## Completion Criteria

The subproject is complete when:

- every physical subsystem uses adjacent, separate ket and bra TT cores;
- no physical TT core has local dimension `d_j^2`;
- Hamiltonian and HEOM generators agree with their dense references in focused
  tests;
- product mixed states and correlated physical twin states are supported;
- root observables can be evaluated without dense expansion;
- all focused and existing TTDynamics tests pass;
- the public API and tensor ordering are documented.

## Non-Goals

- Implementing ChargeTransportModels itself in this subproject.
- Automatically deriving bath correlations from explicit harmonic or
  anharmonic modes.
- Adding TT-Redfield dynamics.
- Replacing or changing the layout of existing HEOM-TT system types.
- Supporting an all-ket-then-all-bra core ordering.
- Inferring a globally Hermitian Hamiltonian by expanding a large MPO.
