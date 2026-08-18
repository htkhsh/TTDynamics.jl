"""
    save_tt_binary(path, tt; overwrite=false) -> String

Write a tensor-train tensor or matrix to the versioned TTDynamics binary format.
"""
function save_tt_binary end

"""
    load_tt_binary(path) -> Union{TTTensor, TTMatrix}

Load a tensor-train tensor or matrix from the versioned TTDynamics binary format.
"""
function load_tt_binary end

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

function _write_tt(io::IO, tt::Union{TTTensor,TTMatrix})
    T = eltype(first(tt.cores))
    haskey(_TT_SCALAR_CODES, T) || throw(ArgumentError("unsupported TT scalar type: $T"))

    write(io, _TT_BINARY_MAGIC)
    _write_le(io, _TT_BINARY_VERSION)
    write(io, _object_kind(tt))
    write(io, _TT_SCALAR_CODES[T])
    _write_le(io, UInt32(length(tt.cores)))
    for core in tt.cores
        _write_core(io, core)
    end
end

function _read_tt(io::IO)
    try
        magic = read(io, length(_TT_BINARY_MAGIC))
        length(magic) == length(_TT_BINARY_MAGIC) && magic == _TT_BINARY_MAGIC ||
            _format_error("invalid TT binary magic")
        _read_le(io, UInt16) == _TT_BINARY_VERSION ||
            _format_error("unsupported TT binary version")

        kind = read(io, UInt8)
        kind in (_TT_TENSOR_KIND, _TT_MATRIX_KIND) ||
            _format_error("invalid TT binary object kind")

        scalar_code = read(io, UInt8)
        haskey(_TT_CODE_TYPES, scalar_code) ||
            _format_error("invalid TT binary scalar code")
        T = _TT_CODE_TYPES[scalar_code]

        core_count = _read_le(io, UInt32)
        iszero(core_count) && _format_error("core count must be positive")
        ndims = _core_ndims(kind)
        cores = [
            _read_core(io, T,
                       _checked_dimensions(ntuple(_ -> _read_le(io, UInt64), ndims), index))
            for index in 1:Int(core_count)
        ]
        rank_dimension = kind == _TT_TENSOR_KIND ? 3 : 4
        for index in 2:length(cores)
            size(cores[index - 1], rank_dimension) == size(cores[index], 1) ||
                _format_error("invalid TT ranks: core $index has a mismatched left rank")
        end

        tt = try
            kind == _TT_TENSOR_KIND ? TTTensor(cores) : TTMatrix(cores)
        catch error
            message = sprint(showerror, error)
            occursin("rank mismatch", message) || rethrow()
            _format_error("invalid TT ranks: $message")
        end
        eof(io) || _format_error("trailing bytes")
        tt
    catch error
        error isa EOFError && _format_error("unexpected end of file")
        rethrow()
    end
end

function _validate_tt_for_save(tt::Union{TTTensor,TTMatrix})
    cores = tt.cores
    isempty(cores) && throw(ArgumentError("TT cores must be non-empty"))
    length(cores) > typemax(UInt32) &&
        throw(ArgumentError("TT core count exceeds UInt32 capacity"))

    T = eltype(first(cores))
    haskey(_TT_SCALAR_CODES, T) || throw(ArgumentError("unsupported TT scalar type: $T"))
    for (core_index, core) in pairs(cores)
        for dimension in size(core)
            dimension > 0 ||
                throw(ArgumentError("TT core $core_index has a zero dimension"))
            UInt128(dimension) > typemax(UInt64) &&
                throw(ArgumentError("TT core $core_index dimension exceeds UInt64 capacity"))
        end
    end
    nothing
end

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

function load_tt_binary(path::AbstractString)
    open(String(path), "r") do io
        _read_tt(io)
    end
end
