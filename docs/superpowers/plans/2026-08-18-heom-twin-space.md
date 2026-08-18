# HEOM-TT Twin-Space Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the standard single `N^2` vectorized HEOM system core with explicit `ket(N) | bra(N)` TT cores throughout TTDynamics.

**Architecture:** `src/heom.jl` builds every Hamiltonian, decay, upward, and downward HEOM term directly from local ket, bra, and hierarchy factors. Public builders retain their signatures but return `[N,N,nb...]` tensors/operators; new dimension and root-density helpers provide canonical validation and diagnostics.

**Tech Stack:** Julia 1.11+, TTDynamics, TTSolver `TTTensor`/`TTMatrix`, KaisouEOM `NoiseExp`/`BathExp`, LinearAlgebra, Test.

## Global Constraints

- Standard HEOM mode order is exactly `[ket(N), bra(N), hierarchy_1(nb[1]), ...]`.
- The old `[N^2, nb...]` representation is removed; no compatibility keyword or automatic conversion is added.
- Left action `A rho` places `A` on ket; right action `rho B` places `transpose(B)` on bra.
- Right action `rho V'` therefore places `conj(V)` on bra.
- No dense `N^2 x N^2` system superoperator may be formed in HEOM construction.
- `HEOMTTSystem`, `build_heom_liouvillian`, and `build_initial_state` public call signatures remain valid.
- `build_heom_liouvillian` continues to return `(liouvillian, trace_observable, population_observables)`.
- Existing vectorized HEOM binary states must be rejected clearly and regenerated.
- Generic TT binary compatibility is unchanged.
- Tests use complex nonsymmetric density matrices and complex operators; real-symmetric-only evidence is insufficient.
- Long Holstein production trajectories and the paused current-correlation workflow are out of scope.

---

## File Structure

- Modify `src/heom.jl`: system validation, twin dimensions, local-factor TT builders, initial state, observables, root-density extraction, and factorized Liouvillian.
- Modify `src/TTDynamics.jl`: export `heom_tt_dimensions` and `root_density_matrix`.
- Create `test/heom_twin_space.jl`: focused complex-algebra, state, observable, validation, dense-reference, and short-propagation tests.
- Modify `test/runtests.jl`: include the focused twin-space tests and update old dimension assumptions.
- Modify `docs/src/api.md`: document twin layout, right-action transpose, new helpers, and incompatibility.
- Modify `examples/heom/sb_ohmic_heomtt.jl`: update printed/assumed TT layout if present.
- Modify `examples/holstein/holstein_brownian_heomtt.jl`: report twin system cores and avoid `N^2` assumptions.
- Modify `examples/holstein/README.md`: state that old vectorized checkpoints must be regenerated.

### Task 1: Twin-Space State, Observables, Dimensions, and Validation

**Files:**
- Modify: `src/heom.jl`
- Modify: `src/TTDynamics.jl`
- Create: `test/heom_twin_space.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `HEOMTTSystem`, `TTTensor`, `TTMatrix`, `NoiseExp`, `BathExp`.
- Produces: `heom_tt_dimensions(system::HEOMTTSystem)::Vector{Int}`.
- Produces: `root_density_matrix(state::TTTensor, system::HEOMTTSystem)::Matrix{ComplexF64}`.
- Produces private `_build_twin_initial_state`, `_validate_heom_state`, `_twin_trace_observable`, `_twin_population_observable`, `_local_product_matrix`, and `_local_product_tensor` helpers.
- Defers switching public `build_initial_state` until Task 2 switches the Liouvillian and observables in the same commit.

- [ ] **Step 1: Write failing twin initial-state and dimension tests**

Create `test/heom_twin_space.jl` with imports and a reusable two-state complex
system:

```julia
using LinearAlgebra
using Test
using TTDynamics
using TTSolver
using KaisouEOM

function twin_test_system(; nb=2)
    H = ComplexF64[0.3 0.7 + 0.2im; 0.7 - 0.2im -0.4]
    V = ComplexF64[0.6 0.1 + 0.3im; 0.1 - 0.3im -0.2]
    bath = BathExp(ComplexF64[0.4 + 0.1im], ComplexF64[0.03 - 0.02im], V)
    HEOMTTSystem(H, NoiseExp([bath]), nb)
end

@testset "HEOM twin-space states" begin
    system = twin_test_system()
    @test heom_tt_dimensions(system) == [2, 2, 2]

    rho = TTDynamics._build_twin_initial_state(system, 2; tol=1e-14)
    @test tt_dims(rho) == [2, 2, 2]
    @test root_density_matrix(rho, system) == ComplexF64[0 0; 0 1]
    @test tt_ranks(rho) == [1, 1, 1, 1]
    @test_throws ArgumentError TTDynamics._build_twin_initial_state(system, 0)
    @test_throws ArgumentError TTDynamics._build_twin_initial_state(system, 3)
end
```

Include `heom_twin_space.jl` in `test/runtests.jl` immediately after
`tt_io.jl`.

- [ ] **Step 2: Run the focused test and verify old dimensions fail**

```bash
julia --project=. test/heom_twin_space.jl
```

Expected: FAIL because the new helpers are undefined and the current initial
state has dimensions `[4,2]`.

- [ ] **Step 3: Harden `HEOMTTSystem` and add canonical dimensions**

Replace constructor assertions with `ArgumentError` validation:

```julia
function HEOMTTSystem(H_sys::AbstractMatrix, noise::NoiseExp, nb::Vector{Int})
    size(H_sys, 1) == size(H_sys, 2) && !isempty(H_sys) ||
        throw(ArgumentError("H_sys must be a nonempty square matrix"))
    all(isfinite, H_sys) || throw(ArgumentError("H_sys must be finite"))
    length(nb) == noise.nterms ||
        throw(ArgumentError("nb length must equal the number of bath terms"))
    all(>(0), nb) || throw(ArgumentError("all hierarchy sizes must be positive"))
    for (index, coupling) in pairs(noise.V)
        size(coupling) == size(H_sys) ||
            throw(ArgumentError("bath coupling $index must match H_sys dimensions"))
        all(isfinite, coupling) ||
            throw(ArgumentError("bath coupling $index must be finite"))
    end
    HEOMTTSystem(ComplexF64.(H_sys), noise, Int.(nb))
end

heom_tt_dimensions(system::HEOMTTSystem) =
    [nsys(system), nsys(system), system.nb...]
```

Ensure the scalar-`nb` constructor delegates to this validated vector
constructor. Add constructor tests for rectangular/empty/nonfinite Hamiltonian,
wrong coupling size, wrong hierarchy count, and nonpositive hierarchy size.

- [ ] **Step 4: Implement rank-one local product helpers and twin initial state**

`_local_product_tensor(vectors)` constructs a rank-one `TTTensor` by reshaping
each vector to `(1,length(vector),1)`. `_local_product_matrix(matrices)` does
the corresponding `(1,n,m,1)` TTMatrix construction and validates each matrix.

Implement `_build_twin_initial_state` with one-hot ket, one-hot bra, then one
vacuum vector per hierarchy mode. Validate the site and round once at the end.
Leave the public `build_initial_state` implementation unchanged in this task so
existing vectorized Liouvillian callers remain internally consistent until the
atomic public switch in Task 2.

- [ ] **Step 5: Implement state-layout validation and root extraction**

`_validate_heom_state` compares `tt_dims(state)` with
`heom_tt_dimensions(system)`. If actual dimensions equal
`[nsys(system)^2, system.nb...]`, throw:

```text
old vectorized HEOM state is unsupported; regenerate it in twin-space format
```

For other mismatches, include expected and actual arrays in `ArgumentError`.

`root_density_matrix` must select hierarchy index one on every auxiliary mode
without expanding the full hierarchy. Contract/slice hierarchy cores with
their one-hot vacuum covectors, retain the possibly entangled ket/bra cores,
and return an `N x N` matrix with entry `[i,j] = rho[i,j]`. Add an entangled
ket/bra test with nontrivial TT rank and nonzero auxiliary entries so the test
proves only the root ADO is returned. Add explicit old-layout and arbitrary
mismatch rejection tests.

- [ ] **Step 6: Build trace and population observables in twin space**

Implement the trace ket core `(1,N,N)` and bra core `(N,N,1)` with matching
diagonal bond entries. Append hierarchy vacuum cores. Population `i` uses
rank-one one-hot ket and bra cores plus hierarchy vacua.

Temporarily expose these only through the existing
`build_heom_liouvillian`; Task 2 completes that builder. For Task 1 tests, call
the private helpers with module qualification or test through a small internal
factory. Construct a complex root matrix and assert trace/population TT dots
equal `tr(rho)` and `diag(rho)`.

- [ ] **Step 7: Export helpers and run tests**

Add `heom_tt_dimensions` and `root_density_matrix` to `src/TTDynamics.jl`.

```bash
julia --project=. test/heom_twin_space.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: focused tests and the full suite pass. The new focused tests exercise
private twin construction directly, while the public builder remains
vectorized only until Task 2.

- [ ] **Step 8: Commit the state representation**

```bash
git add src/heom.jl src/TTDynamics.jl test/heom_twin_space.jl test/runtests.jl
git commit -m "feat: add HEOM twin-space states"
```

### Task 2: Fully Factorized Twin-Space HEOM Liouvillian

**Files:**
- Modify: `src/heom.jl`
- Modify: `test/heom_twin_space.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: Task 1 local-product helpers and `[N,N,nb...]` convention.
- Produces: twin-space `build_heom_liouvillian(system; tol=1e-12)` with unchanged return tuple.
- Switches: public `build_initial_state` to delegate to `_build_twin_initial_state` in the same commit.
- Produces private `_left_factors`, `_right_factors`, and `_hierarchy_factors` or equivalently focused factor-list helpers; no dense `N^2` matrices.

- [ ] **Step 1: Write failing direct left/right action tests**

Add private helper tests using a nonsymmetric complex `rho`, complex `A`, and
complex `B`. Create a twin density tensor whose ket/bra coefficient matrix is
`rho`, apply `_local_product_matrix([A,I])` and
`_local_product_matrix([I,transpose(B)])`, then assert root matrices equal
`A*rho` and `rho*B`. Separately verify right action by `B'` uses `conj(B)` on
bra and equals `rho*B'`.

- [ ] **Step 2: Run the focused test and establish the old builder failure**

```bash
julia --project=. test/heom_twin_space.jl
```

Expected: new factor-action tests fail until helpers/builder are implemented;
the old Liouvillian still reports `[4,4]` system modes instead of `[2,2]`.

- [ ] **Step 3: Implement factor-list construction for every HEOM term**

Build local identity lists in exact mode order:

```julia
[I_system, I_system, hierarchy_identities...]
```

For each product, copy the list, replace only the required ket, bra, or
hierarchy factor, and pass it to `_local_product_matrix`. Implement:

- `G_k`: `gamma[k] * n_k` on hierarchy core `k`;
- upward left: `sqrt_abs_c * V_k` on ket and `bminus_k`;
- upward right: `sqrt_abs_c * transpose(V_k)` on bra and `bminus_k`;
- downward left: `(c1_k/sqrt_abs_c) * V_k` on ket and `bplus_k`;
- downward right: `(c2_k/sqrt_abs_c) * conj(V_k)` on bra and `bplus_k`;
- Hamiltonian left: `H` on ket;
- Hamiltonian right: `transpose(H)` on bra.

Assemble signs exactly as the current equation:

```text
L = -im*(H_left - H_right) - G
    -im*(up_left - up_right)
    -im*(down_left - down_right)
```

Retain coefficient thresholds and periodic/final `tt_round`. Delete all
construction of `Is_super`, `Vx_k`, `L_sys_mat`, and other `N^2 x N^2`
matrices.

In this same change, replace public `build_initial_state` with a delegation to
Task 1's `_build_twin_initial_state`, and return Task 1's twin trace/population
observables from `build_heom_liouvillian`. This is the single atomic public
representation switch.

- [ ] **Step 4: Build an independent dense twin-space reference**

In the test file, independently assemble the dense operator with Julia `kron`
in the same factor order, without calling production factor helpers. For a
two-state, one-term, hierarchy-size-two system, explicitly create `n`, `bplus`,
and `bminus` and evaluate the formula above. Assert:

```julia
liouvillian, trace_observable, populations =
    build_heom_liouvillian(system; tol=1e-14)
@test tt_dims(liouvillian) == ([2,2,2], [2,2,2])
@test tt_full(liouvillian) ≈ dense_reference rtol=1e-12 atol=1e-12
```

Use a complex Hermitian coupling so `transpose(V)` and `conj(V)` differ and a
test cannot pass with the wrong operation.

- [ ] **Step 5: Verify observables, trace null relation, and short propagation**

Assert trace/population observables have `[N,N,nb...]` modes and reproduce a
complex root matrix. Check the dense left null relation
`vec_or_tensor_trace' * tt_full(liouvillian)` to construction tolerance.

Update the existing `Holstein multi-bath HEOM-TT initial state` test to assert
new dimensions. Run one small `tt_ksl` or existing tAMEn short step and assert
root trace conservation within its numerical tolerance.

- [ ] **Step 6: Prove no dense system superoperator remains**

Review `src/heom.jl` and add a structural regression check or targeted source
scan in the test only if robust. At minimum, ensure no `Ns^2 x Ns^2`,
`kron(H_sys_val, Is)`, `kron(Is, ...)`, `Vx_k`, or `L_sys_mat` construction
remains. The factorized dense test is the behavioral proof.

- [ ] **Step 7: Run focused/full tests and commit**

```bash
julia --project=. test/heom_twin_space.jl
julia --project=. -e 'using Pkg; Pkg.test()'
git diff --check
git add src/heom.jl test/heom_twin_space.jl test/runtests.jl
git commit -m "refactor: factorize HEOM in twin space"
```

Expected: all focused and full tests pass.

### Task 3: Examples, API Documentation, and Migration Verification

**Files:**
- Modify: `docs/src/api.md`
- Modify: `examples/heom/sb_ohmic_heomtt.jl`
- Modify: `examples/holstein/holstein_brownian_heomtt.jl`
- Modify: `examples/holstein/README.md`
- Modify: `test/heom_twin_space.jl`

**Interfaces:**
- Consumes: completed twin-space public API.
- Produces: accurate user documentation and import-safe migrated examples.

- [ ] **Step 1: Add failing example-layout/import assertions**

Add tests or guarded subprocess checks that include/import HEOM examples
without running long trajectories and verify any constructed small system uses
`heom_tt_dimensions`. Do not execute the default Holstein 100 fs propagation.
If the examples expose only guarded `main`, test helper construction with a
small config or use source-level assertions limited to stale `[N^2,...]`
phrases.

- [ ] **Step 2: Run checks and identify stale vectorized assumptions**

```bash
rg -n 'N[sS]?\^2|nsys²|vectorized|vec\(rho|vec\(ρ|system core' src/heom.jl docs/src/api.md examples/heom examples/holstein
julia --project=. test/heom_twin_space.jl
```

Expected before edits: API docs and/or examples still describe the old
vectorized system core.

- [ ] **Step 3: Update API documentation**

Document:

- exact `[ket(N), bra(N), hierarchy...]` mode order;
- `A rho` on ket and `rho B` through `transpose(B)` on bra;
- trace observable's ket-bra rank `N`;
- `heom_tt_dimensions` and `root_density_matrix` signatures;
- unchanged builder signatures and changed TT dimensions; and
- explicit incompatibility with old saved `[N^2,nb...]` HEOM states.

Remove or correct every old formula that claims a single vectorized system
core.

- [ ] **Step 4: Migrate examples and user-visible diagnostics**

Update `examples/heom/sb_ohmic_heomtt.jl` and the Holstein builder/printing so
rank/core counts acknowledge two system cores. Use
`heom_tt_dimensions(problem.system)` rather than spelling dimensions. Keep
program guards and numerical defaults unchanged.

Add a Holstein README migration note:

```text
HEOM-TT checkpoints created before twin-space-v1 used one N^2 system core and
must be regenerated; generic non-HEOM TT binary files are unaffected.
```

- [ ] **Step 5: Add binary reload compatibility tests**

Save/load a new twin-space initial state with `save_tt_binary` and assert exact
cores plus valid `root_density_matrix`. Save/load a synthetically constructed
old-layout tensor and assert `_validate_heom_state`/`root_density_matrix`
rejects it with the migration message. This verifies the binary format remains
generic while HEOM semantic validation is strict.

- [ ] **Step 6: Run final verification**

```bash
git diff --check
julia --project=. test/heom_twin_space.jl
julia --project=. -e 'using Pkg; Pkg.test()'
rg -n 'N[sS]?\^2|nsys²|vectorized|vec\(rho|vec\(ρ' src/heom.jl docs/src/api.md examples/heom examples/holstein
```

Expected: formatting clean; focused/full tests pass; remaining search hits, if
any, occur only in explicit migration text or unrelated dense examples and are
reviewed individually.

- [ ] **Step 7: Commit and audit scope**

```bash
git add docs/src/api.md examples/heom/sb_ohmic_heomtt.jl examples/holstein/holstein_brownian_heomtt.jl examples/holstein/README.md test/heom_twin_space.jl
git commit -m "docs: migrate HEOM examples to twin space"
git status --short
git log --oneline -3
```

Confirm no generated CSV/PNG/binary/Manifest files, paused current-correlation
implementation, or unrelated TFD files are committed. Run focused and full
tests once more from the committed tree and record exact pass counts.
