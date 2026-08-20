# HEOM-TT Alignment with KaisouEOM Conjugate Modes

## Objective

Make the HEOM-TT Liouvillian in `TTDynamics` reproduce the exponential-mode
semantics and Liouvillian action of `KaisouEOM`. In particular, exponential
terms whose decay exponents form a complex-conjugate pair must use the
pair-aware `c1`, `c2`, and `abs_coeff` values produced by
`KaisouEOM.NoiseExp`.

This work treats `KaisouEOM` as the reference implementation. It does not
reinterpret or replace its conjugate-pair transformation.

## Scope

The implementation may change the HEOM-TT construction in
`src/heom.jl` and its focused tests. It may add small internal helpers where
they make the mapping auditable.

The following are out of scope:

- changing `KaisouEOM.BathExp` or `KaisouEOM.NoiseExp`;
- changing QFiND TPSD or the Brownian spectral density;
- changing the Holstein Hamiltonian, current operator, physical parameters,
  equilibrium preparation, or output formats;
- changing the tensor layout `[ket, bra, hierarchy...]`;
- replacing the rectangular TT hierarchy with KaisouEOM's total-depth sparse
  hierarchy.

## Reference Semantics

`BathExp` represents

\[
C(t)=\sum_k c_k e^{-\gamma_k t}.
\]

`NoiseExp` expands and pairs the modes. For a complex-conjugate exponent pair,
the resulting hierarchy modes are not reconstructed independently in
`TTDynamics`; their `gamma`, `c1`, `c2`, and `abs_coeff` values are consumed
exactly as provided by `NoiseExp`.

For hierarchy occupation \(n_k\), KaisouEOM uses

\[
\begin{aligned}
\phi_k(n_k) &= \sqrt{(n_k+1)\,\lvert c_{1k}+c_{2k}\rvert},\\
\theta_{Lk}(n_k) &= \sqrt{n_k/\lvert c_{1k}+c_{2k}\rvert}\,c_{1k},\\
\theta_{Rk}(n_k) &= -\sqrt{n_k/\lvert c_{1k}+c_{2k}\rvert}\,c_{2k}.
\end{aligned}
\]

The corresponding action is

\[
\begin{aligned}
\dot\rho_{\mathbf n}={}&-i[H,\rho_{\mathbf n}]
-\sum_k n_k\gamma_k\rho_{\mathbf n}\\
&-i\sum_k\phi_k(n_k)[V_k,\rho_{\mathbf n+\mathbf e_k}]\\
&-i\sum_k\sqrt{n_k/\lvert c_{1k}+c_{2k}\rvert}
\left(c_{1k}V_k\rho_{\mathbf n-\mathbf e_k}
-c_{2k}\rho_{\mathbf n-\mathbf e_k}V_k^\dagger\right).
\end{aligned}
\]

The TT matrix must encode this action with Julia's column-major/twin-space
convention. Ket-core multiplication represents left multiplication. Bra-core
matrices therefore use the transpose required for right multiplication; the
choice of transpose or conjugation must be derived from the reference
\(V_k^\dagger\) action and verified with a nonsymmetric complex coupling
matrix.

## Design

### Noise ownership

`HEOMTTSystem` continues to own a fully constructed `NoiseExp`. The TT builder
must not search for conjugate partners or recompute paired coefficients. This
keeps a single source of truth for real modes, paired complex modes, unpaired
complex modes, sorting, and mode count.

### Auditable local factors

The HEOM-TT builder will express the four contributions using the same
quantities and names as KaisouEOM:

- damping from `gamma` and the hierarchy number operator;
- forward/upward connection from `sqrt(abs_coeff)` and the lowering shift in
  the matrix representation;
- backward-left connection from `c1 / sqrt(abs_coeff)` and the raising shift;
- backward-right connection from `-c2 / sqrt(abs_coeff)` and the raising
  shift.

Zero `abs_coeff` modes must not be divided by zero. Their connection terms are
omitted exactly when their physical coefficients vanish; inconsistent input
with zero scale and nonzero `c1` or `c2` is rejected with an informative
argument error.

### Tensor hierarchy boundary

TTDynamics retains a Cartesian hierarchy with local occupation
`0:(nb[k]-1)`. Connections leaving that box are zero. Reference comparisons
will use the same Cartesian basis and boundary, rather than conflating this
representation choice with KaisouEOM's total-depth hierarchy enumeration.

## Verification

Tests will construct small systems and compare the dense expansion of the TT
Liouvillian against a direct Cartesian implementation of the equations above.
The direct implementation will obtain all mode data from a real
`KaisouEOM.NoiseExp` object.

Required fixtures are:

1. a real exponent;
2. an unpaired complex exponent, for which `NoiseExp` creates its conjugate
   partner;
3. an explicitly supplied complex-conjugate exponent pair with unequal,
   complex coefficients;
4. a nonsymmetric complex coupling matrix, so transpose, conjugation, and
   adjoint mistakes cannot pass accidentally;
5. at least two hierarchy modes and nonuniform local hierarchy sizes, to
   verify mode ordering and boundaries.

For each fixture, compare both the full dense operator and its action on a
generic complex state. Tests must also verify that TTDynamics preserves the
`NoiseExp` mode count, order, `gamma`, `c1`, `c2`, and `abs_coeff` without a
second conjugate-pair transformation.

After the focused tests pass, run the full TTDynamics test suite. A short
Holstein diagnostic may compare the reconstructed problem's mode metadata and
short-time current correlation, but regenerating the 1000 fs equilibrium
artifact is not part of the automated test suite.

## Success Criteria

- Complex-conjugate modes receive exactly the special treatment defined by
  `KaisouEOM.NoiseExp`.
- The TT Liouvillian agrees numerically with the KaisouEOM equations for every
  required fixture and Cartesian hierarchy state.
- Existing twin-space extraction, observables, binary I/O, and Holstein tests
  remain passing.
- No physical parameter or spectral-density convention is changed.

