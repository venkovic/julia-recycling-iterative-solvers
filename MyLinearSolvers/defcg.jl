"""
defcg(A, b, x, W)

Performs Deflated-CG (Saad et al., 2000)

Saad, Y.; Yeung, M.; Erhel, J. & Guyomarc'h, F.
Deflated Version of the Conjugate Gradient Algorithm,
SIAM Journal on Scientific Computing, SIAM, 1999, 21, 1909-1926.

# Examples
```jldoctest
using LinearAlgebra: SymTridiagonal;
using SparseArrays: sparse, SparseMatrixCSC;
using Arpack: eigs;
push!(LOAD_PATH, "./MyLinearSolvers");
using MyLinearSolvers: defcg, cg;
using Random: seed!
seed!(1234);
const n = 100_000;
const T = Float64;
A = sparse(SymTridiagonal(2 .+ .5 * rand(T, n), -1 .+ .05 * rand(T, n-1)));
A = A * A;
function mrhs_defcg(A::SparseMatrixCSC{T}, nvec::Int, nsmp::Int)
  _, W = eigs(A; nev=nvec, which=:SM);
  for ismp in 1:nsmp
    b = rand(T, n);
    _, itdefcg, _ = defcg(A, b, zeros(T, n), W);
    _, itcg, _ = cg(A, b, zeros(T, n));
    println("Def-CG: ", itdefcg, ", CG: ", itcg);
  end
end
nsmp, nvec = 20, 15;
mrhs_defcg(A, nvec, nsmp);
```
"""
function defcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, W::Array{T,2})
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
  mu = WtAW \ (WtA * r)
  p = r - (W * mu)
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
    mu = WtAW \ (WtA * r)
    p = beta * p + r - (W * mu)
    it += 1
    res_norm[it] = sqrt(rTr)
  end
  return x, it, res_norm[1:it]
end

"""
eigdefcg(A, b, x, W, spdim)

Performs RR-LO-TR-Def-CG (Venkovic et al., 2020)

Venkovic, N.; Mycek, P; Giraud, L.; Le Maître, O.
Recycling Krylov subspace strategiesfor sequences of sampled stochastic
elliptic equations,
SIAM Journal on Scientific Computing, SIAM, 2020, under review.

# Examples
```jldoctest
using LinearAlgebra: SymTridiagonal;
using SparseArrays: sparse, SparseMatrixCSC;
using Arpack: eigs;
push!(LOAD_PATH, "./MyLinearSolvers");
using MyLinearSolvers: eigcg, eigdefcg, defcg, cg;
using Random: seed!
seed!(1234);
const n = 100_000;
const T = Float64;
A = sparse(SymTridiagonal(2 .+ .5 * rand(T, n), -1 .+ .05 * rand(T, n-1)));
A = A * A;
function mrhs_eigdefcg(A::SparseMatrixCSC{T}, nvec::Int, nsmp::Int, spdim::Int)
  _, U = eigs(A; nev=nvec, which=:SM);
  W = Array{T}(undef, (n, nvec));
  for ismp in 1:nsmp
    b = rand(T, n);
    if ismp == 1
      _, iteigdefcg, _, W = eigcg(A, b, zeros(T, n), nvec, spdim);
    else
      _, iteigdefcg, _, W = eigdefcg(A, b, zeros(T, n), W, spdim);
    end
    _, itdefcg, _ = defcg(A, b, zeros(T, n), U);
    _, itcg, _ = cg(A, b, zeros(T, n));
    println("eigDef-CG: ", iteigdefcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
  end
end
nsmp, nvec, spdim = 20, 15, 50;
mrhs_eigdefcg(A, nvec, nsmp, spdim);
```
"""
function eigdefcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, W::Array{T,2}, spdim::Int)
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
  n = size(x)[1]
  nvec = size(W)[2]
  nev = nvec
  V = Array{T}(undef, (n, spdim))
  VtAV = zeros(T, spdim, spdim)
  Y = zeros(T, (spdim, 2 * nvec))
  first_restart = true
  #
  it = 1
  r[1:end] = b - A * x
  rTr = dot(r, r)
  mu = WtAW \ (WtA * r)
  p = r - (W * mu)
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
    mu = WtAW \ (WtA * r)
    p = beta * p + r - (W * mu)
    it += 1
    res_norm[it] = sqrt(rTr)
    #
    VtAV[ivec, ivec] += 1 / alpha
    #
    if ivec == spdim
      if first_restart
        VtAV[1:nvec, nvec+1:spdim] = WtA * V[:, nvec+1:spdim]
        first_restart = false
      end
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
    else
      ivec += 1
      V[:, ivec] = r / res_norm[it]
      VtAV[ivec - 1, ivec] = - sqrt(beta) / alpha
      VtAV[ivec, ivec] = beta / alpha
    end
  end
  return x, it, res_norm[1:it], V[:, 1:nvec]
end
