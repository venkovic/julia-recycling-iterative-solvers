push!(LOAD_PATH, "./MyRecyclingKrylovSolvers")
push!(LOAD_PATH, "./MyPreconditioners")

using Random: seed!
using NPZ: npzwrite
using LinearAlgebra: SymTridiagonal, I
using SparseArrays: sprand, sparse
using MyRecyclingKrylovSolvers: eigpcg, initpcg
using MyPreconditioners: BJPreconditioner, BJop, AMGPreconditioner
using MyPreconditioners: Chol16Preconditioner, Chol32Preconditioner

seed!(1234)
const n = 1_000_000
const T = Float64
const nsmp = 10
const nincr = 3

function get_sparse_SPD_mat(n::Int)
  A = sparse(SymTridiagonal(2 .- .05 * rand(T, n), -1 .+ .05 * rand(T, n-1)))
  A = A * A;
  return A
end

const A = get_sparse_SPD_mat(n)
println("\nDone generating A.")

const nblocks = (10, 20, 30)
MbJ = Vector{BJop}(undef, length(nblocks))
for (ibJ, nblock) in enumerate(nblocks)
  MbJ[ibJ] = BJPreconditioner(nblock, A)
  println("Done computing bJ preconditioner for bJ = ", nblock, ".")
end
#
# The AMG preconditioner does not perform as well as it should.
# But A was properly converted into a CSR before building the preconditioner,
# see AMGPreconditioner.jl. Where is the problem ?
#Mamg = AMGPreconditioner(A)
#_, it, _ = pcg(A, b, zeros(T, n), Mamg)
#
#MChol32 = Chol32Preconditioner(A)
#_, it, _ = pcg(A, b, zeros(T, n), MChol32)

const maxit = 2_000
its = Vector{Int}(undef, nsmp)
res_norms = Vector{T}(undef, maxit * nsmp)

const spdim = 70

for (ibJ, nblock) in enumerate(nblocks)
  it_sum = 0
  seed!(4321)
  nvec = nblock
  U = Array{T}(undef, (n, nincr * nvec))
  H = Array{T}(undef, (nincr * nvec, nincr * nvec))
  for ismp in 1:nsmp
    b = rand(T, n)
    if ismp <= nincr
      sl1 = 1 : (ismp - 1) * nvec
      sl2 = (ismp - 1) * nvec + 1 : ismp * nvec
      if ismp == 1
        x = zeros(T, n)
      else
        x = U[:, sl1] * (H[sl1, sl1] \ (U[:, sl1]' * b))
      end
      _, it, res_norm, U[:, sl2] = eigpcg(A, b, x, MbJ[ibJ], nvec, spdim)
      WtA = U[:, sl2]' * A
      H[sl2, sl2] = WtA * U[:, sl2]
      if ismp > 1
        H[sl2, sl1] = WtA * U[:, sl1]
        H[sl1, sl2] = H[sl2, sl1]'
      end
    else
      _, it, res_norm = initpcg(A, b, zeros(T, n), MbJ[ibJ], U)
    end
    Δit = minimum((it, maxit))
    its[ismp] = Δit
    res_norms[(it_sum + 1):(it_sum + Δit)] = res_norm[1:Δit]
    it_sum += Δit
    #
    println("ibJ = $ibJ, nblock = $nblock, it = $it")
  end
  #
  npzwrite("data/mrhs-bJ$nblock-incr-eigpcg.res_norms.npy", res_norms[1:it_sum])
  npzwrite("data/mrhs-bJ$nblock-incr-eigpcg.its.npy", its)
end
