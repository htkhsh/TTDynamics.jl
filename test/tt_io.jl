using TTDynamics
using TTSolver
using Test

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

    @testset "version 1 header" begin
        mktempdir() do directory
            path = joinpath(directory, "header.ttbin")
            save_tt_binary(path, TTTensor([reshape(Float32[1], 1, 1, 1)]))
            bytes = read(path)
            @test bytes[1:8] == UInt8[codeunits("TTDYNBIN")...]
            @test bytes[9:10] == UInt8[0x01, 0x00]
            @test bytes[11:14] == UInt8[0x01, 0x01, 0x01, 0x00]
        end
    end

    @testset "version 1 complex scalar bytes" begin
        mktempdir() do directory
            path = joinpath(directory, "complex-scalar.ttbin")
            tt = TTTensor([reshape(ComplexF32[1.5f0-2.25f0im], 1, 1, 1)])
            save_tt_binary(path, tt)

            bytes = read(path)
            @test bytes[41:44] == reinterpret(UInt8, [htol(reinterpret(UInt32, 1.5f0))])
            @test bytes[45:48] == reinterpret(UInt8, [htol(reinterpret(UInt32, -2.25f0))])
            @test load_tt_binary(path) isa TTTensor{ComplexF32}
        end
    end

    @testset "one object per file" begin
        mktempdir() do directory
            first_path = joinpath(directory, "first.ttbin")
            second_path = joinpath(directory, "second.ttbin")
            first = TTTensor([reshape(Float32[1], 1, 1, 1)])
            second = TTTensor([reshape(Float32[2], 1, 1, 1)])
            save_tt_binary(first_path, first)
            save_tt_binary(second_path, second)
            open(first_path, "a") do io
                write(io, read(second_path))
            end

            @test_throws ArgumentError load_tt_binary(first_path)
        end
    end

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

            directory_target = joinpath(directory, "directory.ttbin")
            mkdir(directory_target)
            marker = joinpath(directory_target, "marker")
            write(marker, "preserve")
            siblings_before_failure = sort(readdir(directory))
            @test_throws Base.IOError save_tt_binary(directory_target, tt; overwrite=true)
            @test isdir(directory_target)
            @test read(marker, String) == "preserve"
            @test sort(readdir(directory)) == siblings_before_failure
        end
    end

    @testset "runtime atomic rename" begin
        mktempdir() do directory
            source = joinpath(directory, "source")
            target = joinpath(directory, "target")
            write(source, "new")
            write(target, "old")

            @test TTDynamics._atomic_rename(source, target) === nothing
            @test !ispath(source)
            @test read(target, String) == "new"

            directory_target = joinpath(directory, "directory")
            mkdir(directory_target)
            write(source, "preserved")
            @test_throws Base.IOError TTDynamics._atomic_rename(source, directory_target)
            @test read(source, String) == "preserved"
            @test isdir(directory_target)
        end
    end

    @testset "invalid files" begin
        append_u16!(bytes, x) = append!(bytes, reinterpret(UInt8, [htol(UInt16(x))]))
        append_u32!(bytes, x) = append!(bytes, reinterpret(UInt8, [htol(UInt32(x))]))
        append_u64!(bytes, x) = append!(bytes, reinterpret(UInt8, [htol(UInt64(x))]))

        function tensor_file_bytes(core_count; scalar_code=0x02)
            bytes = UInt8[codeunits("TTDYNBIN")...]
            append_u16!(bytes, 1)
            push!(bytes, 0x01, scalar_code)
            append_u32!(bytes, core_count)
            bytes
        end

        function append_tensor_core!(bytes, dimensions, element_count)
            for dimension in dimensions
                append_u64!(bytes, dimension)
            end
            for _ in 1:element_count
                append_u64!(bytes, 0)
            end
            bytes
        end

        tt = TTTensor([reshape(Float64[1, 2], 1, 2, 1)])
        mktempdir() do directory
            valid_path = joinpath(directory, "valid.ttbin")
            save_tt_binary(valid_path, tt)
            valid = read(valid_path)

            function rejection(name, bytes)
                path = joinpath(directory, name)
                write(path, bytes)
                try
                    load_tt_binary(path)
                catch error
                    @test error isa ArgumentError
                    return error
                end
                @test false
            end

            function rejected(name, bytes)
                rejection(name, bytes)
            end

            bad_magic = copy(valid); bad_magic[1] = 0xff
            bad_version = copy(valid); bad_version[9:10] .= UInt8[0x02, 0x00]
            bad_kind = copy(valid); bad_kind[11] = 0xff
            bad_scalar = copy(valid); bad_scalar[12] = 0xff
            zero_cores = copy(valid); zero_cores[13:16] .= 0x00

            rejected("magic.ttbin", bad_magic)
            rejected("truncated-magic.ttbin", valid[1:7])
            rejected("version.ttbin", bad_version)
            rejected("kind.ttbin", bad_kind)
            rejected("scalar.ttbin", bad_scalar)
            rejected("cores.ttbin", zero_cores)
            rejected("truncated-header.ttbin", valid[1:12])
            rejected("truncated-data.ttbin", valid[1:end-1])
            rejected("trailing.ttbin", [valid; 0xff])

            zero_dimension = tensor_file_bytes(1)
            append_tensor_core!(zero_dimension, (0, 2, 1), 0)
            rejected("zero-dimension.ttbin", zero_dimension)

            huge_dimension = tensor_file_bytes(1)
            append_tensor_core!(huge_dimension, (UInt64(typemax(Int)) + 1, 1, 1), 0)
            rejected("huge-dimension.ttbin", huge_dimension)

            dimension_product_overflow = tensor_file_bytes(1)
            append_tensor_core!(
                dimension_product_overflow, (UInt64(typemax(Int)), 2, 1), 0)
            error = rejection("dimension-product-overflow.ttbin", dimension_product_overflow)
            @test occursin("core 1 element count overflows Int", sprint(showerror, error))

            payload_byte_overflow = tensor_file_bytes(1; scalar_code=0x04)
            append_tensor_core!(
                payload_byte_overflow, (1, UInt64(typemax(Int) ÷ 8), 1), 0)
            error = rejection("payload-byte-overflow.ttbin", payload_byte_overflow)
            @test occursin("core 1 payload byte count overflows Int", sprint(showerror, error))

            header_only_huge_payload = tensor_file_bytes(1)
            append_tensor_core!(
                header_only_huge_payload, (1, UInt64(typemax(Int) ÷ 16), 1), 0)
            error = rejection("header-only-huge-payload.ttbin", header_only_huge_payload)
            message = sprint(showerror, error)
            @test occursin("core 1 payload", message)
            @test occursin("bytes remain", message)

            rank_mismatch = tensor_file_bytes(2)
            append_tensor_core!(rank_mismatch, (1, 2, 2), 4)
            append_tensor_core!(rank_mismatch, (3, 2, 1), 6)
            rejected("rank-mismatch.ttbin", rank_mismatch)

            early_rank_mismatch = tensor_file_bytes(2)
            append_tensor_core!(early_rank_mismatch, (1, 2, 2), 4)
            append_tensor_core!(early_rank_mismatch, (3, 2, 1), 0)
            error = rejection("rank-mismatch-before-data.ttbin", early_rank_mismatch)
            @test occursin("invalid TT ranks", sprint(showerror, error))
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
end
