module MyPreconditioners

using LinearAlgebra: cholesky
using SparseArrays: SparseMatrixCSC
using SuiteSparse.CHOLMOD: Factor
using PyCall: pyimport, PyObject, PyNULL

const pyamg = PyNULL()
const pysparse = PyNULL()

function __init__()
  copy!(pyamg, pyimport("pyamg.aggregation"))
  copy!(pysparse, pyimport("scipy.sparse"))
end

export BJPreconditioner
export Chol32Preconditioner
export Chol16Preconditioner
export AMGPreconditioner

const T = Float64
const T32 = Float32
const T16 = Float16

include("BJPreconditioner.jl")
include("CholPreconditioners.jl")
include("AMGPreconditioner.jl")

end
