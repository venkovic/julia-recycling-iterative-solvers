"""
eigcg(A, b, x, nvec, spdim)

Performs eigCG (Stathopoulos & Orginos, 2010)

Stathopoulos, A. & Orginos, K.
Computing and deflating eigenvalues while solving multiple right-hand side
linear systems with an application to quantum chromodynamics,
SIAM Journal on Scientific Computing, SIAM, 2010, 32, 439-462.

# Examples
```jldoctest
using LinearAlgebra: SymTridiagonal;
using SparseArrays: sparse, SparseMatrixCSC;
push!(LOAD_PATH, "./MyLinearSolvers");
using MyLinearSolvers: eigcg;
using Random: seed!
seed!(1234);
const n = 1_000;
const T = Float64;
A = sparse(SymTridiagonal(2 .+ .5 * rand(T, n), -1 .+ .05 * rand(T, n-1)));
A = A * A;
#
function mrhs_eigcg(A::SparseMatrixCSC{T}, nvec::Int, spdim::Int, nsmp::Int)
  W = Array{T}(undef, (n, nvec));
  b = rand(T, n);
  x = zeros(T, n);
  _, it, _, W = eigcg(A, b, x, nvec, spdim);
  println(it);
  for ismp in 2:nsmp
    b = rand(T, n);
    _, it, _, W = eigcg(A, b, zeros(T, n), nvec, spdim, W);
    println(it);
  end
end
#
nsmp = 10;
nvec, spdim = 8, 30;
mrhs_eigcg(A, nvec, spdim, nsmp);
```
"""
function eigcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, nvec::Int, spdim::Int)
  r, Ap, res_norm = similar(x), similar(x), similar(x)
  #
  n = size(x)[1]
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
  ivec = 1
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
