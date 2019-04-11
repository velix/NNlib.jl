# XNOR if BitArrays -> {T, F} -> {0, 1}
xnor(x::BitArray{1}, y::BitArray{1})::BitArray{1} = x .== y
xnor(x, y) = x .== y

# This implements popcount as incrementing
# by 1 for each ON bit and decrementing for each OFF.
# The builtin compiler popcount just counts ON bits.
# The counting result should be subtracted by the
# length of the argument
popcount(x::BitArray)::Int64 = 2*count(x)-length(x)

function binary_gemm!(col::BitArray{2}, W::BitArray{2}, out::AbstractArray, M, N, K)
    for m in 1:M
        for n in 1:N
            out[m, n] = popcount(xnor(col[m, :], W[:, n]))
        end
    end

    return out

end
