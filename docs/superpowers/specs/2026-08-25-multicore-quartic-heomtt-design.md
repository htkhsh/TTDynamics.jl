# Multi-core HEOM-TT for Quartic Local Oscillators

## Goal

Implement charge transfer in an open tight-binding chain whose electron is
coupled locally to quartic oscillators, with each oscillator damped by an
independent critically damped Brownian bath. The implementation must represent
the electron, every oscillator, and every hierarchy occupation as separate TT
degrees of freedom. It must not construct a global dense system Hamiltonian or
Liouvillian in production calculations.

The reusable multi-core HEOM machinery belongs in `TTDynamics/src/`. The
quartic model, numerical bath correlation, ESPRIT workflow, executable runs,
and model-specific observables belong in `TTDynamics/examples/quartic/`.

The initial deliverable supports stable one-site oscillator relaxation and
two-site electron transfer. A three-site calculation is optional. Counter
terms, dense HEOM propagation, correlated equilibrium preparation, analytic
Brownian decompositions, global total-tier truncation, GPU support, and dynamic
TT-core reordering are outside this scope.

## Reused Infrastructure

- `TTSolver.TTMatrix`, `TTTensor`, `tt_mkron`, TT addition, and `tt_round` build
  and compress operators without expanding the global tensor product.
- `TTSolver.tamen` supplies the existing non-Hermitian-compatible space-time
  solver used by the repository's Crank--Nicolson HEOM examples.
- Existing Holstein examples provide the propagation, progress reporting,
  output, and plotting patterns.
- `ExpFit.esprit` supplies complex ESPRIT pole candidates. The quartic example
  adds stability checks, conjugate closure, independent coefficient fitting,
  and validation diagnostics that `ExpFit.Exponentials` does not retain.
- The current dense-system `HEOMTTSystem` and `build_heom_liouvillian` remain
  unchanged for backward compatibility.

## Physical Model

Use units with \(\hbar=k_B=1\). The one-electron electronic Hilbert space has
dimension \(N\), so the total electron number is one by construction. Open
boundary conditions are used:

\[
H_{\rm el}=\sum_n\epsilon_n |n\rangle\langle n|
-J\sum_{n=1}^{N-1}(|n\rangle\langle n+1|+|n+1\rangle\langle n|).
\]

Every site has one local oscillator,

\[
h_{\rm loc}=\frac{\Omega_p}{2}p^2+\frac{K_2}{2}q^2
+\frac{K_4}{4}q^4,
\qquad
H_{\rm el-loc}=g\sum_n |n\rangle\langle n|q_n.
\]

Require \(\Omega_p>0\), \(K_4\ge 0\), and allow any finite \(K_2\). No bath
counter term is added. Consequently, changing bath strength, decay rate, or
temperature cannot change the system MPO.

Construct each oscillator in a harmonic basis of size `d_raw` at
`basis_frequency`. Diagonalize the quartic Hamiltonian, retain its lowest
`d_keep` eigenvectors, and project `q`, `q^2`, `p`, and `h_loc` into that
basis. In the harmonic limit `K4 = 0` and
`Omega_p = K2 = basis_frequency = Omega0`, the spectrum reproduces
\(\Omega_0(n+1/2)\) up to basis truncation.

## TT Representation and Ordering

The system density is locally vectorized using Julia's column-major `vec`
convention. The quartic example uses these physical Liouville cores:

```text
electron                N^2
oscillator 1            d_keep^2
...
oscillator N            d_keep^2
```

Hierarchy cores follow all physical cores in site-major, mode-minor order:

```text
hierarchy (site 1, pole 1), ..., hierarchy (site 1, pole K),
hierarchy (site 2, pole 1), ..., hierarchy (site N, pole K)
```

Each hierarchy core has local dimension `hierarchy_nmax + 1`. This physical-
then-hierarchy ordering is the public multi-core HEOM convention. The system
MPO, model observables, and root extraction remain straightforward, while a
future compatible extension may introduce explicit permutations.

For a local Hilbert-space operator \(A\), use

\[
A^L=I\otimes A,\qquad A^R=A^T\otimes I
\]

under column-major vectorization. System Hamiltonian terms are assembled as
TT Kronecker products and sums, then rounded. Only the smallest operator tests
may contract the result to a dense matrix.

## Generic Multi-core HEOM API

Add `src/multicore_heom.jl` and export generic facilities from
`src/TTDynamics.jl`. The central immutable types are equivalent to:

```julia
struct CorrelationFitMetadata
    training_absolute_error::Float64
    training_relative_error::Float64
    validation_absolute_error::Float64
    validation_relative_error::Float64
    sampling_interval::Float64
    fit_time_window::Tuple{Float64,Float64}
    singular_values::Vector{Float64}
    vandermonde_condition::Float64
    minimum_pole_separation::Float64
    maximum_coefficient_magnitude::Float64
    pole_corrections::Vector{String}
end

struct ExponentialCorrelation
    rates::Vector{ComplexF64}
    coeff_forward::Vector{ComplexF64}
    coeff_backward::Vector{ComplexF64}
    scales::Vector{Float64}
    metadata::CorrelationFitMetadata
end
```

`MultiCoreHEOMTTSystem` stores physical core dimensions, the physical
Liouvillian MPO, bath coupling specifications, correlations, and hierarchy
cutoffs. A coupling specification identifies a local physical core and gives
its Hilbert-space operator; constructors form its left and right Liouville
actions and validate dimensions. Different baths may use different
correlations, although the quartic example computes one correlation and reuses
it at every site.

The generic API provides equivalents of:

```julia
build_multicore_heom_generator(system; tol)
build_multicore_heom_initial_state(system, physical_density_cores; tol)
root_ado(state, system)
root_expectation(state, system, observable)
multicore_heom_dimensions(system)
```

Constructors reject empty or mismatched dimensions, nonfinite coefficients,
nonpositive hierarchy sizes or scales, and rates whose real parts are
negative beyond the declared numerical tolerance.

## Correlation Sampling and ESPRIT

The only initially implemented spectral density is

\[
J_{\rm cBO}(\omega)=
\frac{4\lambda\Gamma^3\omega}{(\omega^2+\Gamma^2)^2},
\]

behind a model-local `AbstractSpectralDensity` interface. Numerically evaluate

\[
C(t)=\frac{1}{\pi}\int_0^\infty d\omega\,J(\omega)
\left[\coth(\beta\omega/2)\cos(\omega t)-i\sin(\omega t)\right].
\]

The integration code uses a stable small-argument treatment of `coth` and
supports adaptive integration on either a configured finite interval or a
mapped infinite interval. Integration occurs only while preparing the fit.
Uniform training samples are controlled by `dt_fit`, `t_fit_max`, or an
equivalent consistent sample count, plus absolute and relative quadrature
tolerances. Validation uses a distinct grid and time window.

Pass the complex training series to `ExpFit.esprit`, with either an explicit
rank or a singular-value-relative threshold and rank cap. Treat its output as
pole candidates rather than a validated decomposition:

1. Reject poles outside the Nyquist band or clearly outside the unit circle.
2. Convert \(z_k\) to \(\nu_k=-\log(z_k)/\Delta t\).
3. Round a negative real part to zero only when it lies within the configured
   stability tolerance, and record the correction.
4. Close the accepted pole set under complex conjugation without merging
   distinct nearby poles.
5. On that common basis, solve independently for `coeff_forward` from
   \(C(t)\) and `coeff_backward` from \(C^*(t)\), using QR or SVD least squares.
6. Compute training and independent validation errors, singular values,
   Vandermonde condition number, closest pole distance, and coefficient size.
7. Reject a fit that violates configured stability or error limits.

Critical double-pole behavior is represented by nearby simple exponentials on
the finite fit window. It is diagnosed but not converted to a Jordan-block
HEOM and is never merged solely by pole distance.

## Scaled HEOM Generator

For every bath pole, define

\[
s_k=\sqrt{\max(|c_k^{(+)}|,|c_k^{(-)}|,\epsilon_{\rm scale})}.
\]

The multi-core generator implements

\[
\begin{aligned}
\dot{\widetilde\rho}_{\boldsymbol n}={}&
-i[H_S,\widetilde\rho_{\boldsymbol n}]
-\sum_{\alpha k}n_{\alpha k}\nu_k\widetilde\rho_{\boldsymbol n}\\
&-i\sum_{\alpha k}s_k\sqrt{n_{\alpha k}+1}
[V_\alpha,\widetilde\rho_{\boldsymbol n+\boldsymbol e_{\alpha k}}]\\
&-i\sum_{\alpha k}\frac{\sqrt{n_{\alpha k}}}{s_k}
\left(c_k^{(+)}V_\alpha\widetilde\rho_{\boldsymbol n-\boldsymbol e_{\alpha k}}
-c_k^{(-)}\widetilde\rho_{\boldsymbol n-\boldsymbol e_{\alpha k}}V_\alpha\right).
\end{aligned}
\]

Each hierarchy core supplies identity, number, creation, and annihilation
matrices. The generator is the rounded sum of physical, decay, upward, and
downward TT terms. Tests determine the creation/annihilation matrix orientation
from the canonical component equation rather than copying the existing dense-
system implementation.

## Initial State, Propagation, and Measurements

The default physical state is a TT product of an electron localized at
`initial_site` and each bare quartic oscillator's thermal density
\(e^{-\beta h_{\rm loc}}/Z\). Every hierarchy core starts in occupation zero.
The one-site relaxation executable may replace the oscillator density with a
configured nonequilibrium local state.

Propagate by the repository's Crank--Nicolson/tAMEn pattern. Configuration
includes time step, final time, temporal basis size, tAMEn tolerance, operator
and state rounding tolerances, sweep controls, hierarchy cutoff, and an
enforced TT-rank warning threshold. A reached rank threshold is reported; it
does not silently change the requested model.

At every requested observation time record:

- electron site populations and their sum;
- each oscillator's `q`, `q^2`, and bare local energy;
- root-ADO trace and Hermiticity error;
- maximum and mean TT ranks;
- available tAMEn residual or truncation diagnostics.

Nonfinite states or observables are errors. Configured trace and electron-
number deviations produce explicit diagnostics. Output writers refuse to
overwrite existing results unless requested.

## Quartic Example Files

Create a self-contained example environment and the following files:

```text
examples/quartic/Project.toml
examples/quartic/README.md
examples/quartic/config.jl
examples/quartic/oscillator.jl
examples/quartic/correlation.jl
examples/quartic/model.jl
examples/quartic/dynamics.jl
examples/quartic/plotting.jl
examples/quartic/one_site_relaxation.jl
examples/quartic/two_site_transfer.jl
```

`Project.toml` declares TTDynamics, TTSolver, ExpFit, numerical quadrature, and
plotting dependencies without adding example-only packages to the library.
The README documents local development setup, exact run commands, units,
outputs, convergence controls, and the limitations of nearby ESPRIT poles.

The two-site executable compares harmonic versus quartic oscillators and
zero-bath versus finite-bath dynamics, saving populations and diagnostics with
the complete parameter and fit metadata needed to reproduce the run.

## Testing

Add `test/multicore_heom.jl` and `test/quartic.jl`, and include them from
`test/runtests.jl`. Development follows test-driven implementation in this
order:

1. Harmonic spectrum, `d_raw` convergence, parity selection, and Hermiticity
   for a double well.
2. Independence of the system MPO from bath strength, decay rate, and
   temperature, proving that no counter term is present.
3. cBO reorganization-energy normalization, finite `C(0)`, and quadrature
   convergence.
4. Synthetic complex ESPRIT recovery, including nearby poles and unequal
   amplitudes; common-basis reconstruction of both \(C\) and \(C^*\).
5. cBO training and validation error plus rank-, window-, and sampling-
   dependence diagnostics.
6. Dense contraction of the smallest multi-core generator and comparison to
   an independently assembled scaled HEOM component matrix.
7. Zero-bath unitary dynamics, zero electron--oscillator coupling, trace,
   Hermiticity, and electron-number checks.
8. A short one-site propagation and a short two-site charge-transfer smoke
   test, with output schema checks that do not require plotting.

The first failing test is the quartic oscillator harmonic-limit test. The
generic HEOM dense-contraction test is written before generator implementation.

## Delivery Sequence

One-site delivery proceeds through parameter types, local oscillator basis,
system MPO, spectral density, numerical correlation, ESPRIT validation,
generic scaled generator, factorized state, tAMEn propagation, observables,
and the relaxation executable.

Two-site delivery then adds the open-chain electronic MPO, local Holstein
couplings, population observables, harmonic/quartic and bath/no-bath
comparisons, convergence examples, plots, and reproducibility documentation.

Completion requires all tests to pass, both executables to run with documented
quick configurations, the TT generator to agree with the canonical component
equation, and recorded evidence for fit accuracy, trace, Hermiticity, electron
number, hierarchy cutoff, ESPRIT rank, TT tolerance, time step, and TT ranks.
Any remaining close-pole conditioning or convergence limitation must be
reported rather than hidden.
