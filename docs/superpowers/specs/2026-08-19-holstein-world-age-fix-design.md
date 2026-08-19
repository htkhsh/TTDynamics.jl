# Holstein Builder World-Age Fix

## Problem

`examples/holstein_current_correlation/equilibrate.jl` lazily includes
`holstein_brownian_heomtt.jl` from inside a function. On Julia 1.12, the
included builder methods belong to a newer world than the running default
builder, so direct calls can warn and then fail with a world-age `MethodError`.

## Design

Keep the existing lazy loading so that importing the current-correlation
utilities does not require every example-only dependency. After loading, look
up `decompose_brownian_bcf` and `build_holstein_heomtt` dynamically from the
current module and invoke each with `Base.invokelatest`. This applies the
world-age boundary only to the two functions that may have just been defined.
Injected builders used by tests and callers remain unchanged.

## Error Handling

The existing loader remains responsible for defining both builder bindings.
If the included file fails or does not define a builder, normal Julia include
or missing-binding errors propagate; the fix does not hide dependency or model
construction failures.

## Testing

Add a subprocess regression that starts a fresh Julia 1.12 process, includes
`equilibrate.jl`, and invokes the lazy default builders far enough to prove the
newly included methods are callable without a world-age warning or error. Keep
the test computationally small by using a reduced Holstein configuration and
avoid the 1000 fs propagation. Run the focused current-correlation tests and
the complete package test suite afterward.

## Scope

No changes to the HEOM equations, Brownian TPSD decomposition, default physical
parameters, output format, or propagation duration are included.
