# Peierls HEOM-TT current correlation

## Goal

Add a pure Peierls, periodic-chain example with the same two-stage equilibrium
and current-correlation workflow as `examples/holstein`. The current contains
the bath coordinates, so both its source action and its observation must use
non-root auxiliary density operators (ADOs).

## Model and units

For bond `i -> i+1`, define `B_i = |i+1><i| + |i><i+1|` and
`D_i = |i+1><i| - |i><i+1|`, with periodic indices. The electronic Hamiltonian
is `H_el = sum_i epsilon_i |i><i| - t sum_i B_i`. Independent identical baths
couple through `S_i = B_i - B_(i-1)`, hence
`H_int = sum_i S_i X_i = sum_i B_i (X_i - X_(i+1))`.

The particle current omits the common charge-times-lattice-spacing factor:
`J = sum_i (-im D_i) (-t + X_i - X_(i+1))`. Electronic and bath parameters
are converted from inverse centimeters to inverse femtoseconds before building
the HEOM. Its correlation consequently has units `fs^-2`.

## Core API

Expose two reusable TT operators for `HEOMTTSystem`:

- `build_heom_left_action(system, A; tol)` implements `A rho` on every ADO.
- `build_heom_bath_coordinate_left_action(system, bath_index; tol)` implements
  the coordinate of one `NoiseExp` bath on every ADO.

For mode `k` in bath `b`, the latter has an annihilation coefficient
`sqrt(abs_coeff[k])` and a creation coefficient
`c1[k]/sqrt(abs_coeff[k])`. This is the scaled-ADO convention already used by
the HEOM Liouvillian. A bath mode with zero `abs_coeff` cannot be represented
in that convention and raises `ArgumentError`.

The example composes these operators into the extended current. It applies
that operator to the saved equilibrium HEOM state to form the correlation
source, propagates with the same Liouvillian, and constructs the dual
measurement TT once from the extended current and root trace. Thus the first
tier contributes to the measured current without repeating a TT matrix-vector
product at every time step.

## Workflow

`examples/peierls` contains a configuration, model, workflow helpers,
equilibration entry point, correlation entry point, plotting, README, and a
small validation script. Configuration follows the Holstein naming and supports
PSD/TPSD, localized/system-Gibbs initial states, separate tAMEn tolerances,
BLAS thread count, and periodic site energies. Equilibration saves TT state,
metadata, CSV, population/coherence and rank plots. Correlation reloads the
checkpoint and writes complex CCF and rank diagnostics.

## Verification

1. Compare the TT extended operators against independently assembled dense
   matrices for a tiny hierarchy. Check their action on first-tier ADOs.
2. Compare the Peierls coupling and current matrices against direct
   Kronecker-product constructions for a three-site periodic ring with one
   finite oscillator per site, and check that the current is minus the
   derivative of the Hamiltonian with respect to bond phase.
3. Compare the current correlation against the finite oscillator reference
   through 100 fs, reporting numerical error and
   truncation settings. The finite oscillator and HEOM bath have distinct
   truncations, so agreement is assessed with a stated tolerance.

Reference: Takahashi and Borrelli, J. Chem. Phys. 163, 194105 (2025),
https://doi.org/10.1063/5.0300292.
