using LinearAlgebra, IterativeSolvers, Random, SparseArrays, SuiteSparse

Random.seed!(1234);
const n = 100_000
const T = Float64

diag = 2 .+ .05 * rand(T, n)
off_diag = -1 .- .05 * rand(T, n-1)
A = Tridiagonal(off_diag, diag, off_diag)
A = A'A
diag, off_diag = nothing, nothing

const nblocks = 20
nbJ = Int(floor((n // nblocks)))
slice(i::Int) = (i < nbJ) ? (((i - 1 ) * nbJ + 1):(i * nbJ)) : (((i - 1 ) * nbJ + 1):n)
const bJ_precond = map(i -> cholesky(A[slice(i), slice(i)]), 1:nblocks)

#function apply_inv(x::Vector{T}, bJ_precond::Array{SuiteSparse.CHOLMOD.Factor{T}, 1})
function apply_inv(x::Vector{T})
  y = zeros(T, n)
  @simd for i in 1:nblocks
    @inbounds y[slice(i)] = bJ_precond[i] \ x[slice(i)]
  end
  return y
end

const eps = 1e-7
const maxit = 20_000

res_norm = zeros(T, maxit)
function pcg(A::SparseMatrixCSC{T}, x::Vector{T})
  it = 1
  r = b - A * x
  rTr = r'r
  #z = apply_inv(r, bJ_precond)
  z = apply_inv(r)
  rTz = r'z
  #p = copy(r)
  p = copy(z)
  res_norm[it] = sqrt(rTr)
  bnorm = sqrt(b'b)
  tol = eps * bnorm
  while (it < maxit) && (res_norm[it] > tol)
    Ap = A * p
    d = p'Ap
    #alpha = rTr / d
    alpha = rTz / d
    #beta = 1. / rTr
    beta = 1. / rTz
    x += alpha * p
    r -= alpha * Ap
    rTr = r'r
    #z = apply_inv(r, bJ_precond)
    z = apply_inv(r)
    rTz = r'z
    #beta *= rTr
    beta *= rTz
    #p = beta * p + r
    p = beta * p + z
    it += 1
    res_norm[it] = sqrt(rTr)
  end
  return x, it
end

#@time res_A = lobpcg(A, true, 5)
x = rand(T, n)
b = rand(T, n)

@time x, it = pcg(A, x)
