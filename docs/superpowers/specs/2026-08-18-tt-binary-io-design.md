# Versioned Binary I/O for TT Objects

## Goal

Add a dependency-free binary format for saving and loading `TTSolver.TTTensor`
and `TTSolver.TTMatrix` objects. The format must preserve every core exactly,
identify its own version and object type, reject malformed input clearly, and
avoid Julia's version-sensitive object serialization.

This first version stores one TT object per file. Compression, arbitrary
metadata, collections of TT objects, and non-`TTMatrix` operator types are out
of scope.

## Public API

TTDynamics will export two functions:

```julia
save_tt_binary(path::AbstractString, tt::Union{TTTensor, TTMatrix};
               overwrite::Bool=false) -> String

load_tt_binary(path::AbstractString) -> Union{TTTensor, TTMatrix}
```

`save_tt_binary` returns the path as a `String`. It refuses to replace an
existing file unless `overwrite=true`. `load_tt_binary` determines whether to
construct a `TTTensor` or `TTMatrix` from the file header; callers do not supply
the expected object or scalar type.

Version 1 supports `Float32`, `Float64`, `ComplexF32`, and `ComplexF64` cores.
Saving any other scalar type raises `ArgumentError`. This explicit set keeps the
on-disk representation portable and prevents implicit conversion or precision
loss.

## Code Organization

Binary-format constants and all encoding, decoding, and validation helpers will
live in a new `src/tt_io.jl`. `src/TTDynamics.jl` will include that file and
export only `save_tt_binary` and `load_tt_binary`. Internal byte-order and
scalar I/O helpers will remain private.

The API reference will document the functions, supported scalar types,
overwrite behavior, and compatibility contract. The detailed byte layout will
also be documented so that future format versions can be implemented without
guessing version 1 semantics.

## Version 1 File Format

All integer fields and scalar components use little-endian byte order. No
alignment padding is inserted between fields.

| Field | Size | Meaning |
| --- | ---: | --- |
| Magic | 8 bytes | ASCII `TTDYNBIN` |
| Version | `UInt16` | Format version, initially `1` |
| Object kind | `UInt8` | `1` = `TTTensor`, `2` = `TTMatrix` |
| Scalar type | `UInt8` | Stable code for one of the four supported types |
| Core count | `UInt32` | Number of TT cores; must be positive |
| Core records | variable | One record per core, in TT order |

Each `TTTensor` core record contains three `UInt64` dimensions followed by its
elements. Each `TTMatrix` core record contains four `UInt64` dimensions followed
by its elements. Dimensions use the native Julia core order:
`(r_left, n, r_right)` for tensors and `(r_left, n, m, r_right)` for matrices.

Elements are written in Julia column-major order. Real elements are stored as a
single IEEE-754 component. Complex elements are stored as interleaved real and
imaginary IEEE-754 components of the corresponding precision. The scalar type
code therefore determines the exact number of data bytes in every core.

The file ends immediately after the final core element. Extra trailing bytes
are invalid rather than reserved extension space. A future incompatible layout
must use a new version number.

## Write Path and Failure Behavior

Before writing, `save_tt_binary` checks that the object has at least one core,
that every core uses a supported scalar type, and that dimensions and core
counts fit their on-disk integer fields. The existing TT constructors already
guarantee core dimensionality and adjacent-rank consistency for ordinary TT
objects; the writer will nevertheless derive all dimensions from the actual
cores rather than from cached metadata.

The complete file is written to a uniquely named temporary file in the target
directory. After the stream is closed successfully, the temporary file is
moved to the requested path. This prevents a failed write from leaving a
partially written target. If an error occurs first, the temporary file is
removed. With `overwrite=false`, an existing target is rejected before writing;
with `overwrite=true`, only the final successful move may replace it.

I/O failures propagate with their original exception. Invalid arguments, such
as an unsupported scalar type, raise `ArgumentError` with the offending type or
field identified.

## Read Path and Validation

`load_tt_binary` reads the fixed header first and rejects:

- an incorrect or truncated magic value;
- unsupported versions, object-kind codes, or scalar-type codes;
- a zero core count;
- zero dimensions, dimensions that cannot be represented by Julia `Int`, or
  products that overflow while computing an allocation size;
- truncated core data or any trailing bytes.

The loader allocates each core only after validating its dimensions and element
count. It then reads scalar components explicitly in little-endian order and
reconstructs the recorded element type without numerical conversion.

After all cores are read, construction goes through the normal `TTTensor` or
`TTMatrix` constructor. Consequently, mismatched adjacent TT ranks are rejected
by the same invariant checks used elsewhere in TTSolver. Format and validation
failures are reported as `ArgumentError` with enough context to identify the
invalid header or core. Ordinary filesystem and stream errors remain ordinary
I/O exceptions.

The loader never invokes Julia object deserialization and never evaluates data
from the file.

## Compatibility Contract

Readers dispatch on the format version. Version 1 readers accept only version
1 and fail explicitly on newer versions; they do not attempt a best-effort
parse. Future readers must continue to support version 1 or provide a documented
migration path. Object-kind and scalar-type numeric codes are permanent once
published and must not be reassigned.

Exact binary equality between files produced on different machines is expected
for TT objects with identical core values, because byte order, field sizes,
element order, and scalar encodings are fixed by this specification.

## Tests

Automated tests will cover both object kinds with all four supported scalar
types. For every round trip they will compare the loaded concrete object type,
`eltype`, core count, every core's dimensions and values, TT dimensions, and TT
ranks. Values must compare exactly; this format performs no arithmetic or lossy
conversion.

Additional tests will verify:

- refusal to overwrite by default and successful replacement when requested;
- unsupported scalar types on save;
- corrupted and truncated magic, unsupported version, invalid object kind, and
  invalid scalar code;
- zero or overflowing dimensions and truncated element data;
- adjacent-rank mismatch detected during construction;
- rejection of trailing bytes;
- cleanup behavior when a write fails before publication.

The existing package test suite will be run after the focused binary-I/O tests
to detect integration regressions.
