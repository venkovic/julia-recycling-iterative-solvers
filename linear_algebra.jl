using LinearAlgebra, IterativeSolvers, Random, SparseArrays, SuiteSparse, NPZ

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

#function apply_inv_precond(x::Vector{T}, bJ_precond::Array{SuiteSparse.CHOLMOD.Factor{T}, 1})
function apply_inv_precond(x::Vector{T})
  y = zeros(T, n)
  @simd for i in 1:nblocks
    @inbounds y[slice(i)] = bJ_precond[i] \ x[slice(i)]
  end
  return y
end

const eps = 1e-7
const maxit = 20_000

res_norm = zeros(T, maxit)
function pcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T})
  it = 1
  r = b - A * x
  rTr = r'r
  #z = apply_inv_precond(r, bJ_precond)
  z = apply_inv_precond(r)
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
    z = apply_inv_precond(r)
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
#x = rand(T, n)
#b = rand(T, n)
#
#@time x, it = pcg(A, b, x)
#
#npzwrite("A.npz", A)
#npzwrite("res_norm.npy", res_norm[1:it])
#
#using Plots
#plot(1:it, res_norm[1:it], yaxis=:log)
#savefig("pcg.png")

const DoF = 4005
my_pydata = "/home/nicolas/pydata/"
my_mesh_dir = my_pydata * "meshes/"
my_coeff_dir = my_pydata * "2DKL_block_SExp_L0.1/sig21_ublock_4000_wclust20_svd_morton3_100000/"
#
I, J, V = Vector{Float64}, Vector{Float64}, Vector{Float64}
I = npzread(my_mesh_dir * "2D_ublock_4000_M_I.npy") .+ 1
J = npzread(my_mesh_dir * "2D_ublock_4000_M_J.npy") .+ 1
V = npzread(my_mesh_dir * "2D_ublock_4000_M_V.npy")
M = sparse(I, J, V)
I, J, V = nothing, nothing, nothing
#
const t0 = 27_000
const m = 2_000
const nvec = 100
#
function load_reals(DoF::Int, m::Int, t0::Int)
  X = zeros(Float64, DoF, m)
  Xmean = zeros(Float64, DoF)
  Xtmp = zeros(Float64, DoF)
  for t in t0:(t0 + m - 1)
    Xtmp = npzread(my_coeff_dir * "real$t.npy")
    X[:, t - t0 + 1] = Xtmp
    Xmean += Xtmp
  end
  Xmean ./= m
  return X, Xmean
end
#
X, Xmean = load_reals(DoF, m, t0)
#
XtMX = Array{Float64, 2}
@time XtMX = Symmetric(X'M*X ./ (m - 1))
@time evals = eigvals(XtMX)
npzwrite(my_coeff_dir * "evals.npy", evals[m:-1:m - nvec + 1])
#
X .-= Xmean
@time XtMX = Symmetric(X'M*X ./ (m - 1))
@time evals = eigvals(XtMX)
npzwrite(my_coeff_dir * "evals_zeromean.npy", evals[m:-1:m - nvec + 1])
