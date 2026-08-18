# Versioned TT Binary I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe, dependency-free, versioned binary save/load support for `TTTensor` and `TTMatrix` with exact preservation of supported floating-point core data.

**Architecture:** A focused `src/tt_io.jl` owns the complete version-1 wire format, including fixed little-endian primitive I/O, type/object dispatch, core records, validation, and atomic publication. `TTDynamics.jl` exposes only `save_tt_binary` and `load_tt_binary`; tests treat the file header as a public compatibility contract and exercise both valid round trips and hostile input.

**Tech Stack:** Julia 1.11+, TTDynamics, TTSolver `TTTensor`/`TTMatrix`, Julia Base I/O/filesystem APIs, `Test`; no new package dependency.

## Global Constraints

- The magic is exactly the 8 ASCII bytes `TTDYNBIN`, and the format version is `UInt16(1)`.
- Every integer field and floating-point component is encoded little-endian with no padding.
- Object codes are permanently `UInt8(1)` for `TTTensor` and `UInt8(2)` for `TTMatrix`.
- Version 1 supports exactly `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`; save performs no scalar conversion.
- Tensor cores use `(r_left, n, r_right)` and matrix cores use `(r_left, n, m, r_right)`.
- Core elements use Julia column-major order; complex values use interleaved real and imaginary components.
- One file stores exactly one TT object, and any trailing byte is invalid.
- Existing targets are protected unless `overwrite=true`; a failed write must not publish a partial target.
- Julia `Serialization` and new package dependencies are prohibited.

---

## File Structure

- Create `src/tt_io.jl`: format constants, endian-safe primitive I/O, core record encoding/decoding, public save/load functions, validation, and atomic publication.
- Modify `src/TTDynamics.jl`: export the two public functions and include `tt_io.jl` after the existing source files.
- Create `test/tt_io.jl`: isolated round-trip, binary-layout, malformed-input, overwrite, and cleanup tests.
- Modify `test/runtests.jl`: include the focused test file before example-specific tests.
- Modify `docs/src/api.md`: document signatures, supported types, overwrite semantics, format layout, and compatibility behavior.

### Task 1: Exact Round Trips and the Version 1 Encoder/Decoder

**Files:**
- Create: `test/tt_io.jl`
- Create: `src/tt_io.jl`
- Modify: `test/runtests.jl:1-6`
- Modify: `src/TTDynamics.jl:7-49`

**Interfaces:**
- Consumes: `TTSolver.TTTensor`, `TTSolver.TTMatrix`, `tt_dims`, and `tt_ranks`.
- Produces: `save_tt_binary(path::AbstractString, tt::Union{TTTensor,TTMatrix}; overwrite::Bool=false)::String` and `load_tt_binary(path::AbstractString)::Union{TTTensor,TTMatrix}`.
- Produces private format constants `_TT_BINARY_MAGIC`, `_TT_BINARY_VERSION`, object-kind codes, and scalar-type codes used by Task 2 tests.

- [ ] **Step 1: Write failing exact-round-trip tests**

Create `test/tt_io.jl` with deterministic nontrivial ranks and all supported scalar types:

```julia
@testset "TT binary I/O" begin
    @testset "exact round trips" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            values(::Type{S}, n) where {S<:AbstractFloat} = S.(1:n) ./ S(7)
            values(::Type{Complex{S}}, n) where {S<:AbstractFloat} =
                Complex{S}.(S.(1:n) ./ S(7), -S.(n:-1:1) ./ S(11))

            tensor = TTTensor([
                reshape(values(T, 12), 1, 3, 4),
                reshape(values(T, 40), 4, 5, 2),
                reshape(values(T, 4), 2, 2, 1),
            ])
            matrix = TTMatrix([
                reshape(values(T, 24), 1, 2, 3, 4),
                reshape(values(T, 80), 4, 5, 2, 2),
                reshape(values(T, 12), 2, 3, 2, 1),
            ])

            mktempdir() do directory
                for (name, original) in (("tensor", tensor), ("matrix", matrix))
                    path = joinpath(directory, "$name.ttbin")
                    @test save_tt_binary(path, original) == path
                    restored = load_tt_binary(path)
                    @test typeof(restored) === typeof(original)
                    @test eltype(first(restored.cores)) === T
                    @test tt_dims(restored) == tt_dims(original)
                    @test tt_ranks(restored) == tt_ranks(original)
                    @test restored.cores == original.cores
                end
            end
        end
    end
end
```

Add `include("tt_io.jl")` immediately after the empty package testset in
`test/runtests.jl` so the focused tests run before example utility tests.

- [ ] **Step 2: Run the focused test and confirm the API is absent**

Run:

```bash
julia --project=. test/tt_io.jl
```

Expected: FAIL with `UndefVarError: save_tt_binary not defined` (the standalone
test file must begin with `using TTDynamics`, `using TTSolver`, and `using Test`).

- [ ] **Step 3: Add public registration and fixed format primitives**

In `src/TTDynamics.jl`, add `save_tt_binary` and `load_tt_binary` to the export
list and add `include("tt_io.jl")` after `include("heom.jl")`.

Start `src/tt_io.jl` with docstrings and stable constants:

```julia
const _TT_BINARY_MAGIC = UInt8[codeunits("TTDYNBIN")...]
const _TT_BINARY_VERSION = UInt16(1)
const _TT_TENSOR_KIND = UInt8(1)
const _TT_MATRIX_KIND = UInt8(2)

const _TT_SCALAR_CODES = Dict{DataType,UInt8}(
    Float32 => 0x01,
    Float64 => 0x02,
    ComplexF32 => 0x03,
    ComplexF64 => 0x04,
)
const _TT_CODE_TYPES = Dict(code => T for (T, code) in _TT_SCALAR_CODES)

_uint_type(::Type{Float32}) = UInt32
_uint_type(::Type{Float64}) = UInt64

_write_le(io::IO, value::T) where {T<:Unsigned} = write(io, htol(value))
_read_le(io::IO, ::Type{T}) where {T<:Unsigned} = ltoh(read(io, T))

function _write_component(io::IO, value::T) where {T<:AbstractFloat}
    U = _uint_type(T)
    _write_le(io, reinterpret(U, value))
end

function _read_component(io::IO, ::Type{T}) where {T<:AbstractFloat}
    U = _uint_type(T)
    reinterpret(T, _read_le(io, U))
end

_write_scalar(io::IO, value::T) where {T<:AbstractFloat} =
    _write_component(io, value)
function _write_scalar(io::IO, value::Complex{T}) where {T<:AbstractFloat}
    _write_component(io, real(value))
    _write_component(io, imag(value))
end

_read_scalar(io::IO, ::Type{T}) where {T<:AbstractFloat} =
    _read_component(io, T)
_read_scalar(io::IO, ::Type{Complex{T}}) where {T<:AbstractFloat} =
    Complex{T}(_read_component(io, T), _read_component(io, T))
```

Use `_TT_CODE_TYPES` only after checking `haskey`; never let a raw `KeyError`
represent corrupt input.

- [ ] **Step 4: Implement core records and minimal public save/load**

Implement one shared writer and a dimension-count-specific reader:

```julia
_object_kind(::TTTensor) = _TT_TENSOR_KIND
_object_kind(::TTMatrix) = _TT_MATRIX_KIND
_core_ndims(kind::UInt8) = kind == _TT_TENSOR_KIND ? 3 : 4

function _write_core(io::IO, core::Array{T,N}) where {T,N}
    for dimension in size(core)
        _write_le(io, UInt64(dimension))
    end
    for value in core
        _write_scalar(io, value)
    end
end

function _read_core(io::IO, ::Type{T}, dimensions::NTuple{N,Int}) where {T,N}
    core = Array{T,N}(undef, dimensions)
    for index in eachindex(core)
        core[index] = _read_scalar(io, T)
    end
    core
end
```

Implement `_write_tt(io, tt)` to write magic, version, kind, scalar code,
`UInt32(length(tt.cores))`, then each core. Implement `_read_tt(io)` to read
the same fields, read three or four dimensions per core, construct arrays, and
return `TTTensor(cores)` or `TTMatrix(cores)` according to the kind.

For this task, the public functions may use direct `open`; Task 2 replaces the
save path with atomic publication and centralizes malformed-input translation:

```julia
function save_tt_binary(path::AbstractString,
                        tt::Union{TTTensor,TTMatrix}; overwrite::Bool=false)
    target = String(path)
    !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
    open(target, "w") do io
        _write_tt(io, tt)
    end
    target
end

function load_tt_binary(path::AbstractString)
    open(String(path), "r") do io
        _read_tt(io)
    end
end
```

- [ ] **Step 5: Run focused tests and inspect the header bytes**

Run:

```bash
julia --project=. test/tt_io.jl
```

Expected: PASS for eight round trips.

Add a header assertion to the test using a one-core `Float32` tensor:

```julia
bytes = read(path)
@test bytes[1:8] == UInt8[codeunits("TTDYNBIN")...]
@test bytes[9:10] == UInt8[0x01, 0x00]
@test bytes[11:14] == UInt8[0x01, 0x01, 0x01, 0x00]
```

Run the focused test again. Expected: PASS, proving the header is little-endian
and contains tensor kind, Float32 code, and one little-endian `UInt32` core.

- [ ] **Step 6: Commit the working round-trip format**

```bash
git add src/TTDynamics.jl src/tt_io.jl test/runtests.jl test/tt_io.jl
git commit -m "feat: add versioned TT binary round trips"
```

### Task 2: Corrupt-Input Validation and Atomic Writes

**Files:**
- Modify: `test/tt_io.jl`
- Modify: `src/tt_io.jl`

**Interfaces:**
- Consumes: Task 1's public API, constants, scalar helpers, `_write_tt`, and `_read_tt`.
- Produces: `_checked_dimensions`, `_format_error`, and an atomic `save_tt_binary` implementation; public signatures remain unchanged.

- [ ] **Step 1: Add failing overwrite and malformed-file tests**

Append testsets which first create one valid file, then mutate independent
copies of its bytes:

```julia
@testset "overwrite policy" begin
    tt = TTTensor([reshape(Float64[1, 2], 1, 2, 1)])
    mktempdir() do directory
        path = joinpath(directory, "state.ttbin")
        save_tt_binary(path, tt)
        original = read(path)
        @test_throws ArgumentError save_tt_binary(path, tt)
        @test read(path) == original
        @test save_tt_binary(path, tt; overwrite=true) == path
        @test load_tt_binary(path).cores == tt.cores
    end
end

@testset "invalid files" begin
    tt = TTTensor([reshape(Float64[1, 2], 1, 2, 1)])
    mktempdir() do directory
        valid_path = joinpath(directory, "valid.ttbin")
        save_tt_binary(valid_path, tt)
        valid = read(valid_path)

        function rejected(name, bytes)
            path = joinpath(directory, name)
            write(path, bytes)
            @test_throws ArgumentError load_tt_binary(path)
        end

        bad_magic = copy(valid); bad_magic[1] = 0xff
        bad_version = copy(valid); bad_version[9:10] .= UInt8[0x02, 0x00]
        bad_kind = copy(valid); bad_kind[11] = 0xff
        bad_scalar = copy(valid); bad_scalar[12] = 0xff
        zero_cores = copy(valid); zero_cores[13:16] .= 0x00

        rejected("magic.ttbin", bad_magic)
        rejected("version.ttbin", bad_version)
        rejected("kind.ttbin", bad_kind)
        rejected("scalar.ttbin", bad_scalar)
        rejected("cores.ttbin", zero_cores)
        rejected("truncated-header.ttbin", valid[1:12])
        rejected("truncated-data.ttbin", valid[1:end-1])
        rejected("trailing.ttbin", [valid; 0xff])
    end
end

@testset "unsupported save type" begin
    tt = TTTensor([reshape(Int[1, 2], 1, 2, 1)])
    mktempdir() do directory
        path = joinpath(directory, "integer.ttbin")
        @test_throws ArgumentError save_tt_binary(path, tt)
        @test !ispath(path)
        @test isempty(readdir(directory))
    end
end
```

Add byte-level cases for a zero first dimension, a first dimension larger than
`typemax(Int)`, and two cores whose adjacent ranks disagree. Build their bytes
with small test-only little-endian helpers rather than relying on host byte
order:

```julia
append_u64!(bytes, x) = append!(bytes, reinterpret(UInt8, [htol(UInt64(x))]))
```

For the rank mismatch, encode tensor dimensions `(1, 2, 2)` and `(3, 2, 1)`
with the required number of zero `Float64` elements. Each case must throw
`ArgumentError`.

- [ ] **Step 2: Run focused tests and observe validation failures**

Run:

```bash
julia --project=. test/tt_io.jl
```

Expected: FAIL because EOF may escape as `EOFError`, invalid header codes may
escape as `KeyError` or be accepted, dimensions are unchecked, trailing bytes
are accepted, and save publication is not yet atomic.

- [ ] **Step 3: Centralize format errors and checked allocation sizes**

Add:

```julia
_format_error(message) = throw(ArgumentError("invalid TT binary file: $message"))

function _checked_dimensions(raw::NTuple{N,UInt64}, core_index::Int) where {N}
    any(iszero, raw) && _format_error("core $core_index has a zero dimension")
    any(value -> value > UInt64(typemax(Int)), raw) &&
        _format_error("core $core_index has a dimension larger than typemax(Int)")
    dimensions = ntuple(i -> Int(raw[i]), N)
    try
        foldl(Base.checked_mul, dimensions; init=1)
    catch error
        error isa OverflowError || rethrow()
        _format_error("core $core_index element count overflows Int")
    end
    dimensions
end
```

Read the magic with `read(io, length(_TT_BINARY_MAGIC))` and compare its exact
length/content. Validate version, kind, scalar code, and positive core count
before allocating anything. Wrap only `EOFError` raised during format parsing
as `ArgumentError("invalid TT binary file: unexpected end of file")`; do not
translate filesystem errors.

After reading the final core, require `eof(io)`, otherwise call
`_format_error("trailing bytes")`. Catch constructor errors caused by adjacent
rank mismatch and rethrow an `ArgumentError` that names the invalid TT ranks.

- [ ] **Step 4: Validate save input and publish through a sibling temporary file**

Before creating a temporary file, validate the scalar type, nonzero core count,
`UInt32` core-count capacity, positive dimensions, and `UInt64` dimension
capacity. Implement save publication with `mktemp(dirname(abspath(target)))`,
close the returned stream through a `do` block or `try/finally`, and write via
`_write_tt`:

```julia
function save_tt_binary(path::AbstractString,
                        tt::Union{TTTensor,TTMatrix}; overwrite::Bool=false)
    target = String(path)
    _validate_tt_for_save(tt)
    !overwrite && ispath(target) &&
        throw(ArgumentError("target already exists: $target"))

    temporary_path = nothing
    try
        temporary_path, io = mktemp(dirname(abspath(target)))
        try
            _write_tt(io, tt)
        finally
            close(io)
        end
        if !overwrite && ispath(target)
            throw(ArgumentError("target already exists: $target"))
        end
        mv(temporary_path, target; force=overwrite)
        temporary_path = nothing
        target
    finally
        temporary_path === nothing || rm(temporary_path; force=true)
    end
end
```

The second existence check narrows the race window before publication. Keep
the temporary file in the target directory so the final rename stays on one
filesystem. Verify that an unsupported scalar type is rejected before `mktemp`,
which makes the cleanup assertion deterministic.

- [ ] **Step 5: Run focused and package tests**

Run:

```bash
julia --project=. test/tt_io.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: the focused test passes all valid and invalid cases; the full package
test suite passes without changing Holstein or KSL behavior.

- [ ] **Step 6: Commit validation and safe publication**

```bash
git add src/tt_io.jl test/tt_io.jl
git commit -m "feat: validate TT binary files and writes"
```

### Task 3: API Documentation and Final Compatibility Verification

**Files:**
- Modify: `docs/src/api.md:215-241`
- Modify: `test/tt_io.jl`

**Interfaces:**
- Consumes: the completed public save/load API and fixed version-1 layout.
- Produces: user-facing API documentation and regression assertions for exact scalar bytes and cross-object type detection.

- [ ] **Step 1: Add a failing wire-format scalar test**

Add a test that saves `TTTensor([reshape(ComplexF32[1.5f0-2.25f0im], 1,1,1)])`.
The first element begins after the 16-byte header and 24 bytes of dimensions,
so assert:

```julia
bytes = read(path)
@test bytes[41:44] == reinterpret(UInt8, [htol(reinterpret(UInt32, 1.5f0))])
@test bytes[45:48] == reinterpret(UInt8, [htol(reinterpret(UInt32, -2.25f0))])
```

Also load the file and assert `load_tt_binary(path) isa TTTensor{ComplexF32}`.
Run `julia --project=. test/tt_io.jl`; expected PASS if Task 1 implemented the
specified component order, otherwise FAIL and correct `_write_scalar` before
continuing.

- [ ] **Step 2: Document the API and compatibility contract**

Add an `### TT Binary I/O (tt_io.jl)` section under Functions in
`docs/src/api.md` containing both signatures:

```julia
save_tt_binary(path, tt; overwrite=false) -> String
load_tt_binary(path) -> Union{TTTensor, TTMatrix}
```

State that save supports `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`,
refuses replacement by default, and publishes only a completed sibling
temporary file. State that load detects tensor versus matrix automatically and
rejects unknown versions and malformed or trailing content.

Document the version-1 field sequence: 8-byte `TTDYNBIN`, little-endian
`UInt16(1)`, object `UInt8`, scalar `UInt8`, little-endian `UInt32` core count,
then each core's little-endian `UInt64` dimensions and column-major values.
Explicitly state that complex values are interleaved real/imaginary components
and Julia `Serialization` is not used.

- [ ] **Step 3: Run formatting checks and the full verification suite**

Run:

```bash
git diff --check
julia --project=. test/tt_io.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: `git diff --check` emits no output; focused tests pass; the complete
package test suite passes.

- [ ] **Step 4: Review the public surface and changed-file scope**

Run:

```bash
git diff --stat HEAD~2
git diff HEAD~2 -- src/TTDynamics.jl src/tt_io.jl test/runtests.jl test/tt_io.jl docs/src/api.md
git status --short
```

Confirm that only the two requested functions are newly exported, no dependency
was added to `Project.toml`, the numeric codes match the design specification,
and unrelated example files are untouched.

- [ ] **Step 5: Commit documentation and compatibility assertions**

```bash
git add docs/src/api.md test/tt_io.jl
git commit -m "docs: document TT binary format"
```

- [ ] **Step 6: Record final verification evidence**

Run once more from the committed tree:

```bash
git status --short
julia --project=. test/tt_io.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: no task-owned uncommitted files, focused tests pass, and the complete
package test suite passes. Report any pre-existing unrelated untracked files
separately rather than modifying them.
