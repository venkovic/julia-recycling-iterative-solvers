"""
initcg(A, b, x, W)

Performs Init-CG (Erhel & Guyomarc'h, 2000)

Erhel, J. & Guyomarc'h, F.
An augmented conjugate gradient method for solving consecutive symmetric
positive definite linear systems,
SIAM Journal on Matrix Analysis and Applications, SIAM, 2000, 21, 1279-1299.

Giraud, L.; Ruiz, D. & Touhami, A.
A comparative study of iterative solvers exploiting spectral information
for SPD systems,
SIAM Journal on Scientific Computing, SIAM, 2006, 27, 1760-1786.

# Examples
```jldoctest
using LinearAlgebra: SymTridiagonal;
using SparseArrays: sparse, SparseMatrixCSC;
using Arpack: eigs;
push!(LOAD_PATH, "./MyLinearSolvers");
using MyLinearSolvers: initcg, defcg, cg;
using Random: seed!
seed!(1234);
const n = 1_000;
const T = Float64;
A = sparse(SymTridiagonal(2 .+ .5 * rand(T, n), -1 .+ .05 * rand(T, n-1)));
A = A * A;
function mrhs_initcg(A::SparseMatrixCSC{T}, nvec::Int, nsmp::Int)
  _, W = eigs(A; nev=nvec, which=:SM);
  for ismp in 1:nsmp
    b = rand(T, n);
    _, itinitcg, _ = initcg(A, b, zeros(T, n), W);
    _, itdefcg, _ = defcg(A, b, zeros(T, n), W);
    _, itcg, _ = cg(A, b, zeros(T, n));
    println("Init-CG: ", itinitcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
  end
end
nsmp = 20;
nvec = 20;
mrhs_initcg(A, nvec, nsmp);
```
"""
function initcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, W::Array{T,2})
  r, Ap, res_norm = similar(x), similar(x), similar(x)
  #
  WtA = W' * A
  WtAW = WtA * W
  #
  r[1:end] = b - A * x
  mu = W' * r
  mu = WtAW \ mu
  x += W * mu
  #
  it = 1
  r[1:end] = b - A * x
  rTr = dot(r, r)
  p = copy(r)
  res_norm[it] = sqrt(rTr)
  #
  bnorm = norm2(b)
  tol = eps * bnorm
  #
  while (it < A.n) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTr / d
    beta = 1. / rTr
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    beta *= rTr
    axpby!(1, r, beta, p) # p = beta * p + r
    it += 1
    res_norm[it] = sqrt(rTr)
  end
  return x, it, res_norm[1:it]
end

"""
eiginitcg(A, b, x, W, spdim)

Performs eigCG for 2nd, 3rd, ... systems (Stathopoulos & Orginos, 2010)

Stathopoulos, A. & Orginos, K.
Computing and deflating eigenvalues while solving multiple right-hand side
linear systems with an application to quantum chromodynamics,
SIAM Journal on Scientific Computing, SIAM, 2010, 32, 439-462.

# Examples
```jldoctest
using LinearAlgebra: SymTridiagonal;
using SparseArrays: sparse, SparseMatrixCSC;
using Arpack: eigs;
push!(LOAD_PATH, "./MyLinearSolvers");
using MyLinearSolvers: eigcg, eiginitcg, defcg, cg;
using Random: seed!
seed!(1234);
const n = 100_000;
const T = Float64;
A = sparse(SymTridiagonal(2 .+ .5 * rand(T, n), -1 .+ .05 * rand(T, n-1)));
A = A * A;
function mrhs_eiginitcg(A::SparseMatrixCSC{T}, nvec::Int, nsmp::Int, spdim::Int)
  _, U = eigs(A; nev=nvec, which=:SM);
  W = Array{T}(undef, (n, nvec));
  for ismp in 1:nsmp
    b = rand(T, n);
    if ismp == 1
      _, iteiginitcg, _, W = eigcg(A, b, zeros(T, n), nvec, spdim);
    else
      _, iteiginitcg, _, W = eiginitcg(A, b, zeros(T, n), W, spdim);
    end
    _, itdefcg, _ = defcg(A, b, zeros(T, n), U);
    _, itcg, _ = cg(A, b, zeros(T, n));
    println("eigInit-CG: ", iteiginitcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
  end
end
nsmp, nvec, spdim = 20, 15, 50;
mrhs_eiginitcg(A, nvec, nsmp, spdim);
```
"""
function eiginitcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, W::Array{T,2}, spdim::Int)
  r, Ap, res_norm = similar(x), similar(x), similar(x)
  #
  r[1:end] = b - A * x
  tmp = W' * r
  WtAW = W' * (A * W)
  tmp = WtAW \ tmp
  x += W * tmp
  #
  n = size(x)[1]
  nvec = size(W)[2]
  nev = nvec
  V = Array{T}(undef, (n, spdim))
  VtAV = zeros(T, spdim, spdim)
  Y = zeros(T, (spdim, 2 * nvec))
  #
  it = 1
  r[1:end] = b - A * x
  rTr = dot(r, r)
  p = copy(r)
  res_norm[it] = sqrt(rTr)
  #
  VtAV[1:nvec, 1:nvec] = WtAW
  V[:, 1:nvec] = W
  #
  ivec = nvec + 1
  V[:, ivec] = r / res_norm[it]
  #
  bnorm = norm2(b)
  tol = eps * bnorm
  #
  while (it < A.n) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTr / d
    beta = 1. / rTr
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    beta *= rTr
    axpby!(1, r, beta, p) # p = beta * p + r
    it += 1
    res_norm[it] = sqrt(rTr)
    #
    VtAV[ivec, ivec] += 1 / alpha
    #
    if ivec == spdim
      T = Symmetric(VtAV)
      Y[:, 1:nvec] = eigvecs(T)[:, 1:nvec]
      Y[1:spdim-1, nvec+1:end] = eigvecs(T[1:spdim-1, 1:spdim-1])[:, 1:nvec]
      nev = rank(Y)
      Q = svd(Y).U[:, 1:nev]
      H = Q' * (T * Q)
      vals, Z = eigen(H)
      V[:, 1:nev] = V * (Q * Z)
      #
      ivec = nev + 1
      V[:, ivec] = r / res_norm[it]
      VtAV[:, :] .= 0
      VtAV[1:nev, 1:nev] = diagm(vals)
      VtAV[ivec, ivec] = beta / alpha
      VtAV[1:nev, ivec] = V[:, 1:nev]' * (A * V[:, ivec])
    else
      ivec += 1
      V[:, ivec] = r / res_norm[it]
      VtAV[ivec - 1, ivec] = - sqrt(beta) / alpha
      VtAV[ivec, ivec] = beta / alpha
    end
  end
  return x, it, res_norm[1:it], V[:, 1:nvec]
end
