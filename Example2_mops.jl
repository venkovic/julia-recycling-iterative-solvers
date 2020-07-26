push!(LOAD_PATH, "./MyRecyclingKrylovSolvers")
push!(LOAD_PATH, "./MyPreconditioners")

using Random: seed!
using NPZ: npzwrite
using LinearAlgebra: SymTridiagonal, I
using SparseArrays: sprand, sparse
using MyRecyclingKrylovSolvers: eigdefpcg, eigpcg
using MyPreconditioners: BJPreconditioner, BJop, AMGPreconditioner
using MyPreconditioners: Chol16Preconditioner, Chol32Preconditioner

seed!(1234)
const n = 1_000_000
const T = Float64
const nsmp = 10

function get_sparse_SPD_mat(n::Int)
  A = sparse(SymTridiagonal(2 .- .05 * rand(T, n), -1 .+ .05 * rand(T, n-1)))
  A = A * A;
  return A
end

const b = rand(T, n)

const nblocks = (10, 20, 30)

const maxit = 2_000
its = Vector{Int}(undef, nsmp)
res_norms = Vector{T}(undef, maxit * nsmp)

const spdim = 70

for (ibJ, nblock) in enumerate(nblocks)
  it_sum = 0
  seed!(4321)
  W = Array{T}(undef, (n, nblock))
  A = get_sparse_SPD_mat(n)
  M = BJPreconditioner(nblock, A)
  println("Done computing bJ preconditioner for bJ = ", nblock, ".")
  #M = AMGPreconditioner(A)
  #M = Chol32Preconditioner(A)
  for ismp in 1:nsmp
    if ismp == 1
      _, it, res_norm, W = eigpcg(A, b, zeros(T, n), M, nblock, spdim)
    else
      _, it, res_norm, W = eigdefpcg(A, b, zeros(T, n), M, W, spdim)
    end
    Δit = minimum((it, maxit))
    its[ismp] = Δit
    res_norms[(it_sum + 1):(it_sum + Δit)] = res_norm[1:Δit]
    it_sum += Δit
    #
    println("ibJ = $ibJ, nblock = $nblock, it = $it")
    A += 1e-5 * SymTridiagonal(2 * rand(T, n) .- 1, .5 .- rand(T, n-1))
  end
  #
  npzwrite("data/mops-bJ$nblock-eigdefpcg.res_norms.npy", res_norms[1:it_sum])
  npzwrite("data/mops-bJ$nblock-eigdefpcg.its.npy", its)
end
