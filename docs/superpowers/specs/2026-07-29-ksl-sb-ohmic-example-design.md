# KSL Spin-Boson Ohmic Example Design

## Goal

Add a standalone example that evolves the existing thermofield spin-boson
Ohmic model with TTSolver's homogeneous KSL integrator and compares its system
populations against a HEOMKit HEOM reference.

## Files

- `examples/tfd/ksl/sb_ohmic.jl` contains the executable simulation and plots.
- `examples/tfd/ksl/utils.jl` contains TT-rank and observable helpers used by
  the example.
- `test/runtests.jl` tests the lightweight numerical contracts without running
  the full QFiND/HEOM simulation.

The existing `examples/tfd/sb-ohmic/sb_ohmic.jl` remains unchanged.

## Physical Model

The example uses the same parameters and construction as the existing
`sb-ohmic` example:

- Ohmic power-law exponential spectral density with `s = 1`;
- cutoff frequency `gamma_c = 50 cm^-1`;
- reorganization energy `lambda = 5 cm^-1`;
- temperature `T = 300 K`;
- system bias `epsilon = 0 cm^-1`;
- tunneling coupling `Delta = 20 cm^-1`;
- initial localized system state `|up>`;
- QFiND identification-discretized thermofield bath;
- an ESPRIT-fitted HEOM reference with hierarchy depth 6;
- final time 500 fs and HEOM step 0.5 fs.

The KSL calculation uses the same TFD Hamiltonian returned by `tt_sbham`.
Frequencies and couplings are converted from inverse centimeters to inverse
femtoseconds before constructing the Hamiltonian.

## KSL Evolution

The Schrödinger equation is written in TTSolver's convention as

```text
d psi/dt + A psi = 0,  A = i H.
```

Each step calls:

```julia
u = tt_ksl(
    u,
    1im * H_tt,
    dt_ksl;
    symmetric=true,
    rmax=rmax,
)
```

The example defaults to `dt_ksl = 2 fs` and `rmax = 30`. KSL is
rank-preserving, so the localized rank-one initial state is zero-padded to an
admissible rank profile before the first step.

## Admissible Rank Profile

For physical dimensions `n_1, ..., n_d`, the TT rank vector has length `d+1`
and boundary ranks one. Each internal rank is

```text
r_i = min(rmax, product(n_1:n_i), product(n_(i+1):n_d)).
```

The products are capped during multiplication, rather than formed directly,
so large Hilbert spaces cannot overflow an integer. `rmax` must be positive
and every physical dimension must be positive.

`admissible_tt_ranks(dims, rmax=30)` implements this calculation in
`examples/tfd/ksl/utils.jl`.

## Initial-State Rank Padding

`pad_tt_ranks(x, ranks)` embeds a TT tensor in larger zero-padded cores without
changing its represented dense tensor:

- validate that the requested vector has `length(x.cores) + 1` entries;
- require boundary ranks one;
- require every requested rank to be at least the existing bond rank;
- require every requested rank to satisfy the left/right Hilbert-space bounds;
- allocate zero-filled complex cores of the requested sizes;
- copy each original core into the leading rank blocks.

This padding supplies KSL with rank capacity but does not perturb the localized
initial state.

## Observables and Outputs

`first_site_populations(psi)` contracts all bath sites into a right
environment and returns the populations of the first/system site without
forming the full state vector.

For every KSL time point the example records:

- the two system populations;
- wavefunction norm;
- the complete TT rank vector.

The example writes:

- `ksl_populations.csv`;
- `sb_ohmic_comparison.png`, comparing HEOM and KSL populations;
- `sb_ohmic_norm.png`;
- `sb_ohmic_rank.png`.

The console output reports bath-fit errors, basis sizes, initial rank profile,
stepwise populations, norm, and maximum internal TT rank.

## Error Handling

Utility functions reject nonpositive dimensions, nonpositive `rmax`,
incorrect rank-vector length, nonunit boundary ranks, rank shrinking, and
mathematically inadmissible requested ranks with `ArgumentError`.

The example asserts that the number of discretized frequencies and
coefficients agree before constructing the TFD bath.

## Tests

Lightweight tests include the utility file directly and verify:

1. `admissible_tt_ranks([2, 3, 4], 30) == [1, 2, 4, 1]`;
2. a smaller cap limits internal ranks;
3. invalid dimensions and rank caps throw `ArgumentError`;
4. rank padding produces the requested profile and preserves `tt_full(x)`;
5. invalid padding requests throw `ArgumentError`;
6. `first_site_populations` returns the known populations of a small state;
7. a small Hermitian two-site Hamiltonian evolved with symmetric KSL preserves
   norm.

The full 500 fs QFiND/ESPRIT/HEOM example is not part of the automated test
suite.

## Non-goals

- Refactoring the existing tAMEn/HEOM example.
- Adding rank enrichment during KSL evolution.
- Making QFiND or ExpFit package dependencies of the TTDynamics library.
- Running the full scientific example in CI.
