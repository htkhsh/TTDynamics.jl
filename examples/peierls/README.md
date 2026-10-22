# Periodic Peierls HEOM-TT example

This example computes equilibrium dynamics and the current-current correlation
of a pure Peierls chain. It follows the two-stage workflow of
`examples/holstein`: equilibrate first, then load that state to evaluate the
current correlation.

## Model

For periodic site indices, let `B_i = |i+1><i| + |i><i+1|`. The electronic
Hamiltonian has hopping `-t B_i`. Independent identical Drude baths couple via
`S_i = B_i - B_(i-1)`, so `H_int = sum_i S_i X_i`. The particle current is

```text
J = sum_i [-im (|i+1><i| - |i><i+1|)] [-t + X_i - X_(i+1)] .
```

The common factor of electronic charge times lattice spacing is omitted. The
current correlation therefore uses `fs^-2` units. The coordinate-dependent
terms act on every ADO. In particular, the measured current reads first-tier
ADOs; a root-only electronic current gives a different correlation.

The formula follows Takahashi and Borrelli,
[J. Chem. Phys. 163, 194105 (2025)](https://doi.org/10.1063/5.0300292),
especially their current and extended-current expressions. The implementation
uses HEOMKit's scaled `NoiseExp` hierarchy.

## Run

From the TTDynamics repository root, prepare the example environment once
using the local sibling packages:

```bash
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=examples/peierls -e 'using Pkg; Pkg.develop([PackageSpec(path="."), PackageSpec(path="../TTSolver"), PackageSpec(path="../HEOMKit"), PackageSpec(path="../QFiND")]); Pkg.instantiate()'
```

Then run:

```bash
julia --project=examples/peierls examples/peierls/equilibrate.jl
julia --project=examples/peierls examples/peierls/current_correlation.jl
julia --project=examples/peierls examples/peierls/validate_dense.jl
```

`config.jl` contains all model and numerical settings. The defaults use a
five-site periodic ring, 400 cm⁻¹ hopping, a Drude cutoff of 50 cm⁻¹,
150 cm⁻¹ reorganization energy, and 300 K. `decomposition_method` can be
`:psd` or `:tpsd`. Set `initial_condition` to `:localized` or `:gibbs` for the
initial *electronic* density; both are subsequently equilibrated with the bath.
The two stages have separate tAMEn tolerances. Crank–Nicolson requires an odd
`temporal_basis_size` of at least three.

The equilibration stage writes a TT checkpoint, TOML metadata, a CSV with
populations/coherences and trace error, plus population/coherence and rank
plots under `output/`. The correlation stage checks that the checkpoint's
physical settings match `config.jl`, then writes complex CCF and rank results.
Progress and final ranks are printed during both stages.

## Independent check

`validate_dense.jl` compares three methods for a separate three-site benchmark:
a dense reference with one finite harmonic oscillator per site, HEOM, and a
thermofield-double (TFD) calculation. All three start from the Gibbs density
of the electronic Hamiltonian alone; the oscillator bath starts in its thermal
state, while TFD represents that thermal state with vacuum modes. The test
uses a 250 cm⁻¹ oscillator, 20 cm⁻¹ coupling, and a 100 fs window with 1 fs
steps. It verifies the two equivalent Peierls interaction forms and checks the
current against a flux derivative. The TFD modes use local dimension two to
keep the full reference Hamiltonian dense and tractable, so its truncation
error is included in the comparison.

The script reports the maximum HEOM/reference, TFD/reference, and HEOM/TFD
differences. It saves `output/peierls_dense_validation.png` with real and
imaginary parts and logarithmic absolute-error curves. This finite-mode test
checks the current formula and first-tier ADO action; it does not represent the
default Drude bath or the long-time equilibrated state. Converge the hierarchy
size, oscillator basis, time step, bath decomposition, and tAMEn tolerances for
production calculations.

`validate_brownian_tfd.jl` is a separate Brownian-bath comparison and leaves
the production Drude configuration unchanged. Its default check uses a 50
cm⁻¹ Brownian frequency, 50 cm⁻¹ damping, 50 cm⁻¹ reorganization energy, and
300 K for 50 fs. The HEOM-TT hierarchy has local size 4 per exponential term;
the TFD boson basis has size 4 per mode. QFiND's ID discretization supplies
the thermofield bath modes, and both methods are propagated with tAMEn from the
system-only Gibbs state. The run accepts `duration_fs`,
`reorganization_energy_cm`, `hierarchy_local_size`, and `boson_size` keyword
overrides. The log reports the ID bath-correlation fit error and the maximum
HEOM–TFD CCF difference; the plot is saved as
`output/peierls_brownian_tfd.png`.
