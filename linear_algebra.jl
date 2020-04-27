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
my_coeff_dir = my_pydata * "2DKL_block_SExp_L0.1/sig21_ublock_4000_wclustnc_svd_morton_100000/"
#
I, J, V = Vector{Float64}, Vector{Float64}, Vector{Float64}
I = (npzread(my_mesh_dir * "2D_ublock_4000_M_I.npy") .+ 1)::Vector{Int}
J = (npzread(my_mesh_dir * "2D_ublock_4000_M_J.npy") .+ 1)::Vector{Int}
V = npzread(my_mesh_dir * "2D_ublock_4000_M_V.npy")::Vector{Float64}
M = sparse(I, J, V)
I, J, V = nothing, nothing, nothing
#
const n_smp = 100_000
const n_clust = 400 # n_clust ∈ {25, 50, 100, 200, 400}
kmeans_labels = (npzread(my_coeff_dir * "wclust$(n_clust)_kmeans_labels.npy") .+ 1)::Vector{Int}
#
function load_reals_of_kmeans_by_clust(DoF::Int, n_smp::Int, i_clust::Int, n_clust::Int)
  m = Int(floor(1.25 * n_smp / n_clust))
  X = zeros(Float64, DoF, m)
  Xmean = zeros(Float64, DoF)
  Xtmp = zeros(Float64, DoF)
  t_clust = 0
  for i_real in 1:n_smp
    if kmeans_labels[i_real] == i_clust
      t_clust += 1
      Xtmp = npzread(my_coeff_dir * "real$(i_real - 1).npy")
      if t_clust <= m
        X[:, t_clust] = Xtmp
      else
        vcat(X, Xtmp)
      end
      Xmean += Xtmp
    end
  end
  Xmean ./= t_clust
  return X[:, 1:t_clust], Xmean, t_clust
end
#
function load_contiguous_seq_of_reals(DoF::Int, m::Int, t0::Int)
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
#
#const t0 = 27_000
#const m = 2_000
#X, Xmean = load_contiguous_seq_of_reals(DoF, m, t0)
#
const nvec = 200
#
XtMX = Array{Float64, 2}
@time for i_clust in 1:n_clust
  println("Analying cluster $(i_clust)/$(n_clust)")
  fname = "wclust$(n_clust)_$(i_clust)_"
  #
  X, Xmean, m = load_reals_of_kmeans_by_clust(DoF, n_smp, i_clust, n_clust)
  XtMX = Symmetric(X'M*X ./ (m - 1))
  evals = eigvals(XtMX)
  npzwrite(my_coeff_dir * fname * "evals.npy", evals[m:-1:m - minimum((nvec, m)) + 1])
  #
  X .-= Xmean
  XtMX = Symmetric(X'M*X ./ (m - 1))
  evals = eigvals(XtMX)
  npzwrite(my_coeff_dir * fname * "evals_zeromean.npy", evals[m:-1:m - minimum((nvec, m)) + 1])
end
