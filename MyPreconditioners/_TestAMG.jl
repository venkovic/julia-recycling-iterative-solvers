push!(LOAD_PATH, ".")

using Random: seed!
using NPZ: npzwrite
using LinearAlgebra: SymTridiagonal, I
using SparseArrays: sprand, sparse
using MyPreconditioners: AMGPreconditioner

seed!(1234)
const n = 1_000
const T = Float64

function get_sparse_SPD_mat(n::Int)
  A = sparse(SymTridiagonal(2 .- .05 * rand(T, n), -1 .+ .05 * rand(T, n-1)))
  A = A * A;
  return A
end

const A = get_sparse_SPD_mat(n)

function save_CSC_sparse_array_for_py(fname::String, n::Int)
  npzwrite(fname * "-" * string(n) * "_colptr.npy", A.colptr)
  npzwrite(fname * "-" * string(n) * "_nzval.npy", A.nzval)
  npzwrite(fname * "-" * string(n) * "_rowval.npy", A.rowval)
end

fname = "random-penta"

save_CSC_sparse_array_for_py(fname, n)

b = rand(T, n)
npzwrite("b.npy", b)

M = AMGPreconditioner(A)
x = M \ b
npzwrite(fname  * "-" * string(n) * "_x.npy", x)
