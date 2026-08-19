# Holstein Example Default Overwrite Design

## Goal

Allow repeated direct execution of the Holstein equilibration and
current-correlation scripts to refresh their output files without failing with
`target already exists`.

## Behavior

`equilibrate_main` and `current_correlation_main` default their `overwrite`
keyword to `true`. Consequently, running either `.jl` file directly replaces
its CSV, binary/TOML state data, and PNG outputs atomically where supported.

Callers can still pass `overwrite=false` to request the existing no-clobber
preflight and rollback behavior. Low-level writers and save helpers retain
`overwrite=false` as their default so library-style use remains conservative.

## Scope

No physical parameters, propagation durations, HEOM-TT algorithms, output
names, schemas, plotting data, or binary formats change. README instructions
state the new executable default and explain the explicit no-clobber option.

## Verification

Injected short-run tests create existing outputs and verify that both main
entry points replace them when `overwrite` is omitted. Existing explicit
`overwrite=false` tests continue to verify collision errors and rollback.
