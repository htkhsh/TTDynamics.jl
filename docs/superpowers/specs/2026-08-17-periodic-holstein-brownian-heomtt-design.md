# Periodic Holstein Brownian HEOM-TT Example Design

## Goal

Add a readable, standalone example under `examples/holstein/` that evolves a
one-dimensional periodic Holstein model in the real-space, one-exciton basis.
Each electronic site is coupled to an independent, identical Brownian bath,
and the reduced density dynamics are propagated with the existing HEOM-TT
implementation and TTSolver's tAMEn integrator.

## Files

- `examples/holstein/utils.jl` contains small, testable model-construction and
  validation helpers.
- `examples/holstein/holstein_brownian_heomtt.jl` contains the executable
  simulation, diagnostics, CSV output, and plots.
- `test/runtests.jl` includes lightweight tests for the model helpers and a
  small HEOM-TT initial-state check.

The full scientific simulation is not run in the automated test suite.

## Physical Model

The electronic subsystem is represented by the site states
`|1>, ..., |N>`. Its Hamiltonian is

```text
H_sys = sum_n epsilon_n |n><n|
        - J sum_n (|n><n+1| + |n+1><n|),
```

with periodic indexing `N + 1 = 1`. The default parameters are:

- site count `N = 5`;
- uniform site energies `epsilon_n = 0 cm^-1`;
- positive hopping magnitude `J = 400 cm^-1`, appearing as `-J` in the
  Hamiltonian;
- initial excitation localized at site 1.

Each site has an independent bath coupled through the projector
`V_n = |n><n|`. All site baths use the same underdamped Brownian spectral
density

```text
J(omega) = 4 lambda Gamma Omega^2 omega
           / ((omega^2 - Omega^2)^2 + 4 Gamma^2 omega^2),
```

with:

- characteristic frequency `Omega = 1400 cm^-1`;
- damping factor `Gamma = 200 cm^-1`;
- reorganization energy `lambda = 600 cm^-1`;
- temperature `T = 300 K`.

Hamiltonian quantities are converted from inverse centimeters to inverse
femtoseconds before propagation.

## Bath-Correlation Expansion

The example constructs `QFiND.BrownianSD(Omega, Gamma, lambda)` and its finite
temperature `BosonicBCF`. The correlation function is sampled on a documented
time grid and fitted to an exponential sum with `ExpFit.esprit`.

The fit quality is reported as a relative 2-norm error. The fitted exponential
parameters are reused for every site, while each `BathExp` receives its own
site projector. `NoiseExp` combines these independent baths into the input
accepted by `HEOMTTSystem`.

## Code Structure

`HolsteinConfig` collects physical parameters, BCF-fitting controls,
hierarchy truncation, tAMEn controls, output cadence, and final time in one
place. Its constructor validates all requirements before expensive work starts.

The utility file exposes focused helpers:

- `periodic_holstein_hamiltonian` builds the Hermitian site Hamiltonian;
- `site_projectors` builds the local system-bath coupling operators;
- `validate_config` rejects invalid physical and numerical inputs.

The executable is divided into functions for:

1. fitting the Brownian correlation function;
2. constructing the multi-bath HEOM-TT problem;
3. propagating one tAMEn step at a time;
4. measuring populations, trace, and TT ranks;
5. writing CSV data and plots;
6. coordinating the run from `main()`.

The script invokes `main()` only when executed directly, so its definitions can
be loaded without starting the expensive simulation.

## HEOM-TT Evolution

The system is created with the existing public API:

```julia
system = HEOMTTSystem(H_sys, noise, hierarchy_local_size)
liouvillian, trace_observable, population_observables =
    build_heom_liouvillian(system)
rho = build_initial_state(system, initial_site)
```

The equation `d rho / dt = L rho` is advanced in fixed-size steps by scaling
the Liouvillian by the step size and applying tAMEn with a Crank-Nicolson time
scheme. After each step, the final snapshot is extracted and rounded with a
separate documented TT tolerance.

The default example evolves from 0 to 100 fs. Time step, hierarchy local size,
tAMEn tolerance, sweep count, local iterations, kick rank, and internal time
grid size remain easy to change in the configuration block. Defaults favor a
useful first run rather than a convergence claim; the comments explicitly tell
users to converge hierarchy size, fit tolerance, time step, and TT tolerance
for production results.

## Observables and Outputs

At every saved time point, the example records:

- all five site populations;
- the physical density-matrix trace;
- maximum and mean TT rank.

Output paths are resolved relative to the script directory rather than the
process working directory. The example creates:

- `holstein_brownian_populations.csv`;
- `holstein_brownian_populations.png`;
- `holstein_brownian_trace.png`;
- `holstein_brownian_rank.png`.

Console output summarizes physical parameters, BCF fit quality, hierarchy
dimensions, initial ranks, periodic progress, trace drift, and output paths.

## Error Handling

Validation rejects:

- fewer than two sites;
- a site-energy vector whose length differs from the site count;
- an initial site outside `1:N`;
- nonpositive temperature, Brownian parameters, hierarchy local size, time
  step, final time, sample count, or solver tolerances;
- a final time that is not an integer multiple of the time step.

The Hamiltonian helper handles `N = 2` without adding the same periodic bond
twice. The script checks that the BCF fit contains at least one stable
exponential and that its relative error is finite before building the HEOM.

## Tests

Lightweight tests verify:

1. the five-site Hamiltonian is Hermitian;
2. every nearest-neighbor matrix element, including the `1 <-> N` bond, is
   `-J` and all non-neighbor off-diagonal elements vanish;
3. the two-site Hamiltonian does not double its single bond;
4. site projectors are Hermitian, mutually orthogonal, and sum to identity;
5. invalid configurations fail early with `ArgumentError`;
6. a small synthetic multi-bath `HEOMTTSystem` produces an initial state with
   unit trace and the expected localized population.

The full Brownian BCF fit and 100 fs propagation remain an executable example,
not a CI requirement.

## Non-goals

- Adding explicit phonon Fock spaces to the system Hilbert space.
- Modeling correlated baths or nonlocal electron-phonon coupling.
- Adding disorder, multiple excitons, exciton-exciton interactions, or
  recombination.
- Refactoring the public HEOM-TT API beyond changes required by a demonstrated
  correctness issue.
- Claiming numerical convergence from the example defaults.
