# HEOM-TT Twin-Space Migration Design

## Goal

Replace TTDynamics' vectorized system-density core with an explicit twin-space
representation throughout the standard HEOM-TT implementation.

```text
old: system density (N^2) | hierarchy 1 | hierarchy 2 | ...
new: ket (N) | bra (N) | hierarchy 1 | hierarchy 2 | ...
```

This is a deliberate, repository-wide breaking change. The old representation
will not remain behind a keyword or compatibility mode. Existing saved
vectorized HEOM-TT states must be regenerated.

The previously designed Holstein equilibration/current-correlation example is
paused until this migration is complete. It will then be redesigned on top of
the twin-space API so that current action is a local ket-core TT operator.

## Representation Convention

The first two TT modes represent the matrix coefficient `rho[i,j]` directly:

- ket mode index `i` is the density matrix row;
- bra mode index `j` is the density matrix column; and
- all hierarchy modes follow in the same order as `HEOMTTSystem.nb`.

For a product operator `|a><b|`, the two system cores contain `a` on the ket
mode and `conj(b)` on the bra mode. The full two-mode tensor therefore has
entries `a[i] * conj(b[j]) = rho[i,j]`; its flattened ordering is an internal
TT detail and is not described as Julia `vec(rho)`.

Matrix actions are defined at the tensor-index level:

```text
A rho  -> A on ket, identity on bra
rho B  -> identity on ket, transpose(B) on bra
```

The transpose is algebraic, not an adjoint. Thus right action by `B'` uses
`transpose(B') = conj(B)` on the bra core. These rules support general complex
Hamiltonians and coupling operators rather than relying on real symmetric
examples.

## Public API and Compatibility

The following public call signatures remain unchanged:

```julia
HEOMTTSystem(H_sys, noise, nb)
build_heom_liouvillian(system; tol=1e-12)
build_initial_state(system, initial_site; tol=1e-12)
```

`build_heom_liouvillian` still returns
`(liouvillian, trace_observable, population_observables)`. Their TT mode sizes
change from `[N^2, nb...]` to `[N, N, nb...]`.

`build_initial_state` still creates `|site><site|` in the hierarchy vacuum,
but now returns a rank-one ket/bra product followed by hierarchy vacuum cores.

The migration adds two public inspection/conversion helpers:

```julia
heom_tt_dimensions(system::HEOMTTSystem) -> Vector{Int}
root_density_matrix(state::TTTensor, system::HEOMTTSystem) -> Matrix{ComplexF64}
```

`heom_tt_dimensions` returns `[N, N, system.nb...]` and provides one canonical
dimension check for examples and state reloaders. `root_density_matrix`
contracts every hierarchy mode with its vacuum basis vector and returns the
ket/bra matrix. It is intended for diagnostics, validation, and interoperability,
not for propagation.

Functions accepting a HEOM state will validate its TT modes. A state whose
modes match the old `[N^2, nb...]` layout raises an `ArgumentError` explicitly
identifying it as the unsupported vectorized HEOM representation. Other mode
mismatches report expected and actual dimensions.

## Factorized Operator Construction

The implementation will build every HEOM term directly as a TT product or sum
of products. It will never form an `N^2 x N^2` dense system superoperator and
then split it.

Private helpers in `src/heom.jl` will have one responsibility each:

- construct system and hierarchy identity matrices;
- place a matrix on the ket core;
- place a transposed right-action matrix on the bra core;
- place a number, creation, or annihilation matrix on one hierarchy core;
- convert a complete list of local factors into a rank-one `TTMatrix`; and
- build the entangled ket/bra trace tensor.

Repeated sums will be TT-rounded at the existing tolerance and at the same
periodic points as the current implementation. No new dense dependency or
alternative TT type is introduced.

## Liouvillian Terms

### System Hamiltonian

The coherent system generator is

```text
-im * (H on ket - transpose(H) on bra).
```

Both products use identity matrices on every hierarchy core.

### Hierarchy Decay

For every bath expansion term `k`, the decay product uses identity on ket and
bra, `gamma[k] * n_k` on hierarchy core `k`, and hierarchy identities
elsewhere. The total decay operator enters with the same negative sign as the
existing HEOM equation.

### Upward Coupling

For coupling operator `V_k`, the commutator product multiplying the hierarchy
annihilation operator is split into:

```text
 V_k on ket, identity on bra
-identity on ket, transpose(V_k) on bra.
```

Both terms carry the existing `sqrt(abs(c_k))` scaling and the annihilation
operator on hierarchy core `k`.

### Downward Coupling

The two products multiplying the hierarchy creation operator are:

```text
(c1_k / sqrt(abs(c_k))) * V_k rho
(c2_k / sqrt(abs(c_k))) * rho V_k'
```

They are represented by `V_k` on ket for the first product and `conj(V_k)` on
bra for the second product, with the existing relative minus sign. Coefficient
threshold behavior remains unchanged.

## States and Observables

### Initial State

For `initial_site = i`, the ket and bra cores are one-hot vectors at `i`, each
with boundary TT rank one. Every hierarchy core is the local vacuum vector.

### Trace Observable

The root-density trace is not a ket/bra product. It is represented exactly as

```text
sum_i <i|_ket tensor <i|_bra.
```

The ket core has shape `(1, N, N)` and the bra core `(N, N, 1)`, with matching
identity entries on their shared rank. Hierarchy cores are vacuum covectors.

### Population Observables

Population `i` is the rank-one ket/bra product `<i| tensor <i|`, followed by
hierarchy vacuum covectors. `tt_dot(population_i, state)` therefore returns
`rho[i,i]` from the root ADO.

## Validation and Errors

`HEOMTTSystem` construction will reject:

- a non-square or empty system Hamiltonian;
- nonfinite Hamiltonian entries;
- a coupling operator whose dimensions differ from the Hamiltonian;
- a nonpositive hierarchy local size; and
- hierarchy-size count inconsistent with the number of bath terms.

Liouvillian construction will retain checks for finite decay rates and
coefficients where available and will validate all local factor dimensions
before TT construction. Initial-state construction retains explicit site-range
validation.

`root_density_matrix` rejects old-layout states, arbitrary dimension
mismatches, and nonfinite reconstructed matrix values. It does not silently
reshape an old vectorized state.

## Existing Examples and Documentation

All repository HEOM-TT examples that inspect dimensions or create states will
be migrated to the twin-space layout. In particular:

- `examples/heom/sb_ohmic_heomtt.jl` will use the unchanged public builders and
  update any dimension/rank assumptions;
- `examples/holstein/holstein_brownian_heomtt.jl` will continue to call the
  standard builders and will report the additional system core; and
- API documentation will describe `[ket | bra | hierarchy...]`, the transpose
  rule for right action, and incompatibility with previously saved HEOM states.

The dense/non-TT example `examples/heom/sb_ohmic.jl` is changed only if it
shares an explicit assumption about the TT state layout. Unrelated TFD examples
are out of scope.

## Testing

Tests will not rely only on real symmetric Hamiltonians or diagonal couplings.

1. A complex Hermitian Hamiltonian and complex coupling matrix will be used to
   compare each factorized ket/bra action with direct dense matrix algebra on a
   nonsymmetric complex density matrix.
2. A small complete HEOM Liouvillian will be expanded densely and compared
   with an independently assembled dense twin-space reference, including
   decay, upward, and both downward terms.
3. Initial states will reconstruct exactly to `|i><i|` and have dimensions
   `[N, N, nb...]`.
4. Trace and all site-population observables will match direct dense root-ADO
   calculations for a complex test state.
5. The left null relation `trace_observable' * liouvillian` will be checked to
   the construction tolerance.
6. `root_density_matrix` will be tested on entangled ket/bra TT ranks and
   nonzero auxiliary ADOs to ensure only the hierarchy vacuum is selected.
7. Old `[N^2, nb...]` states and arbitrary mismatches will be rejected with
   specific errors.
8. Existing HEOM-TT initial-state and short-propagation tests will be updated
   and kept green.
9. The full TTDynamics test suite will run after focused twin-space tests.

The tests will use small system and hierarchy dimensions; no production
Holstein trajectory or long propagation is run.

## Migration Notes

This is a format-level HEOM state migration, not a change to the generic TT
binary format. Generic `TTTensor`/`TTMatrix` files remain readable, but a
reloaded HEOM state must match `heom_tt_dimensions(system)` and any application
metadata must record:

```text
heom_representation = "twin-space-v1"
```

The documentation will instruct users to regenerate old equilibrium or
checkpoint states. Automatic conversion is intentionally omitted because a
bare TT binary file does not identify whether an `N^2` mode represents a
density matrix or an unrelated physical mode.

## Non-Goals

- retaining the old vectorized HEOM implementation;
- automatic conversion of old binary states;
- changing the mathematical hierarchy truncation or bath decomposition;
- changing TTSolver's global core-ordering convention;
- adding a general matrix-product-density-operator package; and
- implementing the paused Holstein current-correlation workflow in this
  migration step.
