"""
defcg(A, b, x, W)

Performs Deflated-CG (Saad et al., 2000).

Used to solve A x = b with an SPD matrix A when a set of linearly independent
vectors w1, w2, ... is known and such that, ideally, Span{w1, w2, ...} is
"approximately" invariant under the action of A. Then, the sequence of iterates
of Def-CG is equivalent to a post-processed sequence of a regular CG solve of a
deflated version of the linear system, with guaranteed decrease of the condition
number. Remark: if Span{w1, w2, ...} is exactly invariant under the action of A,
one should use Init-CG instead of Def-CG as both algorithms should have equally
positive impacts on convergence, but Def-CG entails an additional computational
cost at every solver iteration.

Saad, Y.; Yeung, M.; Erhel, J. & Guyomarc'h, F.
Deflated Version of the Conjugate Gradient Algorithm,
SIAM Journal on Scientific Computing, SIAM, 1999, 21, 1909-1926.

# Examples
```jldoctest
julia>
using LinearAlgebra: SymTridiagonal;
using SparseArrays: sparse, SparseMatrixCSC;
using Arpack: eigs;
push!(LOAD_PATH, "./MyRecyclingKrylovSolvers");
using MyRecyclingKrylovSolvers: defcg, cg;
using Random: seed!
seed!(1234);
const n = 1_000;
const T = Float64;
A = sparse(SymTridiagonal(2 .+ .5 * rand(T, n), -1 .+ .05 * rand(T, n-1)));
A = A * A;
function mrhs_defcg(A::SparseMatrixCSC{T}, nvec::Int, nsmp::Int)
  _, W = eigs(A; nev=nvec, which=:SM);
  println("\\n* Def-CG results *");
  for ismp in 1:nsmp
    b = rand(T, n);
    defcg(A, b, zeros(T, n), W);
    _, itdefcg, _ = defcg(A, b, zeros(T, n), W);
    _, itcg, _ = cg(A, b, zeros(T, n));
    println("Def-CG: ", itdefcg, ", CG: ", itcg);
  end
end
nsmp, nvec = 5, 20;
mrhs_defcg(A, nvec, nsmp);

* Def-CG results *
Def-CG: 140, CG: 183
Def-CG: 141, CG: 183
Def-CG: 141, CG: 183
Def-CG: 140, CG: 184
Def-CG: 140, CG: 183
```
"""
function defcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, W::Array{T,2})
  r, Ap, res_norm, p = similar(x), similar(x), similar(x), similar(x)
  #
  WtA = W' * A
  WtAW = WtA * W
  #
  if iszero(x)
    r .= b
  else
    r = b - A * x
  end
  mu = W' * r
  mu = WtAW \ mu
  x += W * mu
  #
  it = 1
  r = b - A * x
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

Performs RR-LO-TR-Def-CG (Venkovic et al., 2020), here referred to as eigDef-CG.

Works as a combination of eigCG and Def-CG. The linear solve is deflated as in
Def-CG, and approximate least dominant eigenvectors of A are computed throughout
the solve in a similar way as in eigCG. This can be used as an alternative to
the incremental eigCG algorithm, or for sequences of systems As xs = bs with
correlated SPD matrices ..., As-1, As, As+1. An example is shown for each case.

Venkovic, N.; Mycek, P; Giraud, L.; Le Maître, O.
Recycling Krylov subspace strategiesfor sequences of sampled stochastic
elliptic equations,
SIAM Journal on Scientific Computing, SIAM, 2020, under review.

# Examples
```jldoctest
julia>
using LinearAlgebra: SymTridiagonal;
using SparseArrays: sparse, SparseMatrixCSC;
using Arpack: eigs;
push!(LOAD_PATH, "./MyRecyclingKrylovSolvers");
using MyRecyclingKrylovSolvers: eigcg, eigdefcg, defcg, cg, initcg;
using Random: seed!
seed!(1234);
const n = 1_000;
const T = Float64;
A = sparse(SymTridiagonal(2 .+ .5 * rand(T, n), -1 .+ .05 * rand(T, n-1)));
A = A * A;
nsmp, ndefcg, nvec, spdim = 10, 3, 20, 50;
#
function mrhs_eigdefcg(A::SparseMatrixCSC{T}, nvec::Int, nsmp::Int, spdim::Int, ndefcg::Int)
  _, U = eigs(A; nev=nvec, which=:SM);
  W = Array{T}(undef, (n, nvec));
  println("\\n* eigDef-CG results for multiple right-hand sides *");
  for ismp in 1:nsmp
    b = rand(T, n);
    _, itcg, _ = cg(A, b, zeros(T, n));
    _, itdefcg, _ = defcg(A, b, zeros(T, n), U);
    if ismp == 1
      _, iteigcg, _, W = eigcg(A, b, zeros(T, n), nvec, spdim);
      println("eigCG: ", iteigcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
    else
      if ismp <= ndefcg
        _, iteigdefcg, _, W = eigdefcg(A, b, zeros(T, n), W, spdim);
        println("eigDef-CG: ", iteigdefcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
      else
        _, itinitcg, _ = initcg(A, b, zeros(T, n), W);
        println("Init-CG: ", itinitcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
      end
    end
  end
end
mrhs_eigdefcg(A, nvec, nsmp, spdim, ndefcg);
#
function mops_eigdefcg(A::SparseMatrixCSC{T}, nvec::Int, nsmp::Int, spdim::Int)
  b = rand(T, n);
  W = Array{T}(undef, (n, nvec));
  println("\\n* eigDef-CG results for multiple operators *");
  for ismp in 1:nsmp
    A += sparse(SymTridiagonal(.12 * rand(T, n), -.05 * rand(T, n-1)));
    _, U = eigs(A; nev=nvec, which=:SM);
    _, itcg, _ = cg(A, b, zeros(T, n));
    _, itdefcg, _ = defcg(A, b, zeros(T, n), U);
    if ismp == 1
      _, iteigcg, _, W = eigcg(A, b, zeros(T, n), nvec, spdim);
      println("eigCG: ", iteigcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
    else
      _, iteigdefcg, _, W = eigdefcg(A, b, zeros(T, n), W, spdim);
      println("eigDef-CG: ", iteigdefcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
    end
  end
end
mops_eigdefcg(A, nvec, nsmp, spdim);

* eigDef-CG results for multiple right-hand sides *
eigCG: 183, Def-CG: 140, CG: 183
eigDef-CG: 155, Def-CG: 141, CG: 183
eigDef-CG: 153, Def-CG: 141, CG: 183
Init-CG: 161, Def-CG: 140, CG: 184
Init-CG: 160, Def-CG: 140, CG: 183
Init-CG: 159, Def-CG: 140, CG: 183
Init-CG: 155, Def-CG: 140, CG: 183
Init-CG: 161, Def-CG: 140, CG: 184
Init-CG: 156, Def-CG: 140, CG: 183
Init-CG: 162, Def-CG: 140, CG: 183

* eigDef-CG results for multiple operators *
eigCG: 196, Def-CG: 136, CG: 196
eigDef-CG: 176, Def-CG: 126, CG: 226
eigDef-CG: 152, Def-CG: 122, CG: 253
eigDef-CG: 150, Def-CG: 119, CG: 231
eigDef-CG: 127, Def-CG: 113, CG: 183
eigDef-CG: 113, Def-CG: 111, CG: 160
eigDef-CG: 112, Def-CG: 108, CG: 154
eigDef-CG: 112, Def-CG: 107, CG: 140
eigDef-CG: 111, Def-CG: 100, CG: 137
eigDef-CG: 103, Def-CG: 98, CG: 135
```
"""
function eigdefcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, W::Array{T,2}, spdim::Int)
  r, Ap, res_norm, p = similar(x), similar(x), similar(x), similar(x)
  #
  WtA = W' * A
  WtAW = WtA * W
  #
  if iszero(x)
    r .= b
  else
    r = b - A * x
  end
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
  r = b - A * x
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
      Tm = Symmetric(VtAV) # spdim-by-spdim
      Y[:, 1:nvec] = eigvecs(Tm)[:, 1:nvec] # spdim-by-nvec
      Y[1:spdim-1, nvec+1:end] = eigvecs(Tm[1:spdim-1, 1:spdim-1])[:, 1:nvec] # (spdim-1)-by-nvec
      nev = rank(Y) # nvec <= nev <= 2*nvec
      Q = svd(Y).U[:, 1:nev] # spdim-by-nev
      H = Q' * (Tm * Q) # nev-by-nev
      vals, Z = eigen(H)::Eigen{T,T,Array{T,2},Array{T,1}}
      V[:, 1:nev] = V * (Q * Z) # n-by-nev
      #
      ivec = nev + 1
      V[:, ivec] = r / res_norm[it]
      VtAV .= 0
      for j in 1:nev
        VtAV[j, j] = vals[j]
      end
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

function defpcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, M, W::Array{T,2})
  nothing
end

function eigdefcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, M, W::Array{T,2}, spdim::Int)
  nothing
end
