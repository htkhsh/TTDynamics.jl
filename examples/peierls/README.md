# Periodic Peierls HEOM-TT example

This example computes equilibrium dynamics and the current-current correlation
of a pure Peierls chain. It follows the two-stage workflow of
`examples/holstein`: equilibrate first, then load that state to evaluate the
current correlation.

## Model

For periodic site indices, let `B_i = |i+1><i| + |i><i+1|`. The electronic
Hamiltonian has hopping `-t B_i`. Independent identical baths couple via
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
```

`config.jl` contains all model and numerical settings. The defaults use a
five-site periodic ring, 4000 cm⁻¹ hopping, Brownian frequency 50 cm⁻¹,
damping 90 cm⁻¹, reorganization energy 100 cm⁻¹, and 300 K. Set
`bath_model=:brownian` or `bath_model=:drude` to choose the bath; the Drude
cutoff is configured with `drude_cutoff_cm` (50 cm⁻¹ by default).
`decomposition_method` can be
`:psd` or `:tpsd`. Set `initial_condition` to `:localized` or `:gibbs` for the
initial *electronic* density; both are subsequently equilibrated with the bath.
The two stages have separate tAMEn tolerances. Crank–Nicolson requires an odd
`temporal_basis_size` of at least three.

The equilibration stage writes a TT checkpoint, TOML metadata, a CSV with
populations/coherences and trace error, plus population/coherence and rank
plots under `output/`. The correlation stage checks that the checkpoint's
physical settings match `config.jl`, then writes complex CCF and rank results.
Progress and final ranks are printed during both stages.
Set `output_directory` in `config.jl` to change where these files and the
Brownian TFD validation plot are saved. Relative paths are resolved from
`examples/peierls/`; the default is `"output"`.

## Independent check

`validate_brownian_tfd.jl` compares HEOM with a thermofield-double (TFD)
calculation for a separate Brownian parameter set. Its default check uses a 50
cm⁻¹ Brownian frequency, 50 cm⁻¹ damping, 50 cm⁻¹ reorganization energy, and
300 K for 50 fs. The HEOM-TT hierarchy has local size 4 per exponential term;
the TFD boson basis has size 4 per mode. QFiND's ID discretization supplies
the thermofield bath modes, and both methods are propagated with tAMEn from the
system-only Gibbs state. The run accepts `duration_fs`,
`reorganization_energy_cm`, `hierarchy_local_size`, and `boson_size` keyword
overrides. The log reports the ID bath-correlation fit error and the maximum
HEOM–TFD CCF difference; the plot is saved as
`output/peierls_brownian_tfd.png`.
