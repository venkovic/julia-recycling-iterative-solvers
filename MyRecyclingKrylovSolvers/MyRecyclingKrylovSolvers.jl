module MyRecyclingKrylovSolvers

using LinearAlgebra: mul!, axpy!, axpby!, dot, norm2, iszero, Eigen
using LinearAlgebra: Symmetric, eigvecs, eigen, svd, rank, diagm
using SparseArrays: SparseMatrixCSC

export cg, pcg
export eigcg, eigpcg
export defcg, eigdefcg, defpcg, eigdefpcg
export initcg, initpcg

const T = Float64
const eps = 1e-7

include("cg.jl")
include("eigcg.jl")
include("defcg.jl")
include("initcg.jl")

end
