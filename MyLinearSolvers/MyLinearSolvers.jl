module MyLinearSolvers

using LinearAlgebra: mul!, axpy!, axpby!, dot, norm2
using LinearAlgebra: Symmetric, eigvecs, eigen, svd, rank, diagm
using SparseArrays: SparseMatrixCSC

export cg, pcg
export eigcg
export defcg, eigdefcg
export initcg, eiginitcg

const T = Float64
const eps = 1e-7

include("cg.jl")
include("eigcg.jl")
include("defcg.jl")
include("initcg.jl")

end
