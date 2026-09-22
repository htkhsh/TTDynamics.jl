# tAMEn Conserved-Observable API Design

## Purpose

Add an explicit, representation-independent conservation API to
`TTSolver.tamen`, then use it consistently from TTDynamics propagation for
`HEOMTTSystem`, `MultiCoreHEOMTTSystem`, and `MultiCoreRedfieldTTSystem`.

The immediate physical requirement is trace preservation. For a composite
density matrix, every reduced density matrix has the same trace as the full
density matrix:

```math
\operatorname{Tr}(\operatorname{Tr}_{\bar A}\rho)
= \operatorname{Tr}(\rho).
```

Consequently, MultiCore propagation needs one global trace constraint, not
one independent constraint per physical subsystem. The implementation will
preserve the initial trace, which is normally one but need not be hard-coded
to one.

Conservation will be achieved by retaining the supplied observable directions
in the AMEn approximation spaces. The solver will not rescale or project the
computed state after propagation. Conservation is therefore expected within
the configured tAMEn accuracy, not at forced machine precision.

## Scope

This change covers:

- an additive public keyword API in `TTSolver.tamen`;
- backward compatibility with the existing `opts[:obs]` mechanism;
- shared TTDynamics propagation machinery;
- trace-preserving propagation for standard HEOM, MultiCore HEOM, and
  MultiCore Redfield systems;
- validation, tests, and API documentation.

This change does not preserve subsystem purity, local populations,
Frobenius norms, or the traces of individual higher-tier auxiliary density
operators. It does not add post-step normalization or projection.

## TTSolver API

The public signature will become:

```julia
tamen(
    X,
    A,
    tol,
    opts=Dict{Symbol,Any}(),
    z=nothing;
    conserved_observables=nothing,
    correct_2_norm=nothing,
)
```

`conserved_observables` accepts an array of `TTTensor` instances. Each tensor
represents a linear functional on the spatial state and must have the same
mode dimensions as `X` after excluding its time mode. The conserved value is
the contraction with the initial state. Multiple observables are supported.

The new keyword is a public spelling of the existing auxiliary-observable
mechanism. Internally, validated observables flow through the same AMEn basis
enrichment and initial-component preservation code currently reached through
`opts[:obs]`.

`correct_2_norm=nothing` retains the current option/default behavior. A
non-`nothing` Boolean value supplies the corresponding tAMEn option through
the keyword API.

The existing positional `opts` argument and `opts[:obs]` remain supported.
Supplying both `opts[:obs]` and `conserved_observables` is an
`ArgumentError`. Supplying both `opts[:correct_2_norm]` and a non-`nothing`
`correct_2_norm` keyword is also an `ArgumentError`. Rejecting duplicate
sources avoids hidden precedence rules.

The public docstring will state that:

- observables are linear functionals, not nonlinear norms;
- their initial contractions are the values retained by the algorithm;
- conservation is subject to solver accuracy because no output projection is
  performed;
- `correct_2_norm` is independent and should be disabled for density-matrix
  evolution unless the Frobenius norm is physically conserved.

## TTDynamics API

TTDynamics will expose three propagation functions with consistent keyword
behavior:

```julia
propagate_heom_tamen(
    system::HEOMTTSystem,
    state::TTTensor,
    times;
    preserve_trace=false,
    conserved_observables=nothing,
    options=Dict{Symbol,Any}(),
    ...,
)

propagate_multicore_heom_tamen(
    system::MultiCoreHEOMTTSystem,
    state::TTTensor,
    times;
    preserve_trace=false,
    conserved_observables=nothing,
    options=Dict{Symbol,Any}(),
    ...,
)

propagate_multicore_redfield_tamen(
    system::MultiCoreRedfieldTTSystem,
    state::TTTensor,
    times;
    preserve_trace=false,
    conserved_observables=nothing,
    options=Dict{Symbol,Any}(),
    ...,
)
```

`preserve_trace` defaults to `false` so existing calculations keep their
current ranks and convergence behavior. When it is `true`, TTDynamics adds
the system's trace functional to any explicitly supplied conserved
observables and calls `tamen` with `correct_2_norm=false`.

A caller that requests `preserve_trace=true` while explicitly enabling
2-norm correction receives an `ArgumentError`. Density-matrix dynamics does
not generally preserve the Frobenius 2-norm, and silently applying both
requirements would impose an unphysical constraint. Legacy `options[:obs]`
may still be used when neither new conservation keyword is used; mixing the
legacy and new paths is rejected by the TTSolver validation.

All three functions return one `TTTensor` per requested time, matching the
existing `propagate_multicore_redfield_tamen` contract. They validate a
nonempty finite nondecreasing time grid, compatible initial-state dimensions,
and the existing temporal-basis and tolerance constraints.

## Trace Functionals

### Standard HEOM

The standard HEOM trace functional is the existing twin-space trace
observable. It contracts the ket/bra physical cores with the identity and
selects hierarchy index one on every auxiliary core. It therefore measures
the trace of the root auxiliary density operator.

### MultiCore HEOM

The MultiCore physical layout is

```text
[ket1, bra1, ket2, bra2, ..., hierarchy1, hierarchy2, ...].
```

The conservation functional contracts every physical ket/bra pair with its
local identity and selects hierarchy index one on every hierarchy core. In
symbols, it represents

```math
I_1 \otimes I_2 \otimes \cdots \otimes I_M
\otimes |0\rangle_{\mathrm{hierarchy}}.
```

Its contraction is `Tr(rho_root)`. This equals the trace of every reduced
physical density matrix, so one functional normalizes all subsystem
reductions simultaneously.

The existing `build_multicore_trace_observable` returns only the physical
observable because `multicore_expectation` separately extracts the root ADO.
It will retain that contract. The propagation implementation will add the
hierarchy-vacuum cores through a focused internal helper, producing a tensor
whose dimensions match the complete HEOM state.

### MultiCore Redfield

The Redfield state has only the interleaved physical ket/bra cores. Its trace
functional is the existing MultiCore physical identity observable without
hierarchy-vacuum extension.

## Shared Propagation Implementation

TTDynamics will add `src/tamen_propagation.jl`. It will own:

- common time-grid, tolerance, and state-dimension validation;
- interval-by-interval tAMEn propagation;
- merging an automatically generated trace functional with explicit
  conserved observables;
- trace/2-norm option conflict checks;
- the three public propagation methods.

The existing Redfield propagation body will move out of
`src/multicore_redfield.jl` and use this shared implementation. Generator and
trace-functional construction remain dispatched by system type. Each
nonzero interval builds a short space-time tensor, calls `tamen`, extracts the
endpoint using the time scheme selected in `options` (`"cheb"` by default),
rounds the endpoint, and appends it to the returned trajectory. This removes
the existing mismatch in which Redfield propagation always labels endpoint
extraction as Crank-Nicolson even when `tamen` used its default Chebyshev
scheme. Repeated time values append the unchanged current state.

This organization keeps numerical propagation rules in one unit while
leaving HEOM and Redfield generator construction in their existing modules.

## Error Handling

The implementation reports clear errors for:

- a conserved observable that is not a `TTTensor`;
- observable mode dimensions that do not match the spatial state;
- simultaneous old and new spellings of the same tAMEn option;
- `preserve_trace=true` combined with explicit 2-norm correction;
- a conserved observable incompatible with the selected TTDynamics system;
- invalid time grids, state dimensions, temporal basis sizes, or tolerances.

An empty `conserved_observables` collection is equivalent to no explicit
conservation constraint. Nonfinite or nonpositive solver tolerances continue
to follow the existing propagation validation rules.

## Testing

### TTSolver

A small linear evolution will have a conserved left functional while its
Euclidean 2-norm changes. Tests will verify that:

1. `conserved_observables` retains the initial functional value at extracted
   output times within the configured tAMEn tolerance;
2. disabling 2-norm correction allows the physically expected norm change;
3. both Chebyshev and Crank-Nicolson time schemes support the new API;
4. multiple independent conserved observables are accepted;
5. legacy `opts[:obs]` remains operational;
6. invalid observable types and dimensions are rejected;
7. duplicate legacy/new option specifications are rejected.

### TTDynamics

Focused small-system tests will cover:

1. standard HEOM root-trace preservation;
2. MultiCore HEOM root-trace preservation;
3. unit trace for each explicitly reconstructed reduced density matrix in a
   small MultiCore HEOM case;
4. MultiCore Redfield physical-trace preservation;
5. full HEOM dimension matching for the hierarchy-extended trace functional;
6. unchanged opt-in behavior when `preserve_trace=false`;
7. rejection of trace preservation combined with explicit 2-norm
   correction.

Tests will use deterministic random seeds where tAMEn enrichment randomness
can affect ranks or convergence. Assertions will use tolerances consistent
with the requested solver accuracy and will not claim exact post-normalized
traces.

## Documentation

The TTSolver `tamen` docstring will document the new keywords, compatibility
rules, and accuracy semantics. TTDynamics' API guide will document all three
propagation functions and distinguish root-trace preservation from purity,
population, Frobenius-norm, and higher-tier ADO conservation.

## Success Criteria

The change is complete when:

- existing positional `tamen` and `opts[:obs]` callers still work;
- the new keyword API accepts one or more compatible TT observables;
- each TTDynamics system type offers the same opt-in trace-preservation
  interface;
- standard and MultiCore HEOM preserve the root trace within solver accuracy;
- MultiCore Redfield preserves the physical trace within solver accuracy;
- density-matrix propagation does not apply 2-norm correction when trace
  preservation is enabled;
- no propagation output is rescaled or projected after tAMEn.
