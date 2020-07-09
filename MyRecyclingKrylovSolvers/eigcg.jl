"""
eigcg(A, b, x, nvec, spdim)

Performs eigCG (Stathopoulos & Orginos, 2010).

Used at the begining of a solving procedure of linear systems A xs = bs with a
constant SPD matrix A, and different right-hand sides bs. eigCG may be run once
(or incrementally) to generate approximate least dominant eigenvectors of A.
These approximate eigenvectors are then used to generate a deflated initial
guess with the Init-CG algorithm. Incremental eigCG should be used when the
solve of the first system ends before accurate eigenvector approximations can be
obtained by eigCG, which then limits the potential speed-up obtained for the
subsequent Init-CG solves. See Example for typical use and implementation of the
Incremental eigCG algorithm (Stathopoulos & Orginos, 2010).

Stathopoulos, A. & Orginos, K.
Computing and deflating eigenvalues while solving multiple right-hand side
linear systems with an application to quantum chromodynamics,
SIAM Journal on Scientific Computing, SIAM, 2010, 32, 439-462.

# Examples
```jldoctest
julia>
using LinearAlgebra: SymTridiagonal;
using SparseArrays: sparse, SparseMatrixCSC;
push!(LOAD_PATH, "./MyRecyclingKrylovSolvers");
using MyRecyclingKrylovSolvers: eigcg, initcg, cg;
using Random: seed!
seed!(1234);
const n = 1_000;
const T = Float64;
A = sparse(SymTridiagonal(2 .+ .5 * rand(T, n), -1 .+ .05 * rand(T, n-1)));
A = A * A;
nsmp, neigcg, nvec, spdim = 10, 3, 20, 50;
#
function mrhs_eigcg(A::SparseMatrixCSC{T}, nvec::Int, spdim::Int, nsmp::Int)
  W = Array{T}(undef, (n, nvec));
  println("\\n* eigCG results *");
  for ismp in 1:nsmp
    b = rand(T, n);
    _, itcg, _ = cg(A, b, zeros(T, n));
    if ismp == 1
      _, iteigcg, _, W = eigcg(A, b, zeros(T, n), nvec, spdim);
      println("eigCG: ", iteigcg, ", CG: ", itcg);
    else
      _, itinitcg, _ = initcg(A, b, zeros(T, n), W);
      println("Init-CG: ", itinitcg, ", CG: ", itcg);
    end
  end
end
#
seed!(4321);
mrhs_eigcg(A, nvec, spdim, nsmp);
#
function mrhs_incr_eigcg(A::SparseMatrixCSC{T}, nvec::Int, spdim::Int, nsmp::Int, neigcg::Int)
  U = Array{T}(undef, (n, neigcg * nvec));
  H = Array{T}(undef, (neigcg * nvec, neigcg * nvec));
  println("\\n* Incremental eigCG results *");
  for ismp in 1:nsmp
    b = rand(T, n);
    _, itcg, _ = cg(A, b, zeros(T, n));
    if ismp <= neigcg
      sl1 = 1 : (ismp - 1) * nvec;
      sl2 = (ismp - 1) * nvec + 1 : ismp * nvec;
      if ismp == 1
        x = zeros(T, n);
      else
        x = U[:, sl1] * (H[sl1, sl1] \\ (U[:, sl1]' * b));
      end
      _, iteigcg, _, U[:, sl2] = eigcg(A, b, x, nvec, spdim);
      WtA = U[:, sl2]' * A;
      H[sl2, sl2] = WtA * U[:, sl2];
      if ismp > 1
        H[sl2, sl1] = WtA * U[:, sl1];
        H[sl1, sl2] = H[sl2, sl1]';
      end
      println("eigCG: ", iteigcg, ", CG: ", itcg);
    else
      _, itinitcg, _ = initcg(A, b, zeros(T, n), U);
      println("Init-CG: ", itinitcg, ", CG: ", itcg);
    end
  end
end
#
seed!(4321);
mrhs_incr_eigcg(A, nvec, spdim, nsmp, neigcg);

* eigCG results *
eigCG: 183, CG: 183
Init-CG: 166, CG: 183
Init-CG: 163, CG: 183
Init-CG: 162, CG: 183
Init-CG: 160, CG: 183
Init-CG: 164, CG: 182
Init-CG: 164, CG: 183
Init-CG: 158, CG: 182
Init-CG: 161, CG: 183
Init-CG: 160, CG: 183

* Incremental eigCG results *
eigCG: 183, CG: 183
eigCG: 166, CG: 183
eigCG: 160, CG: 183
Init-CG: 158, CG: 183
Init-CG: 154, CG: 183
Init-CG: 161, CG: 182
Init-CG: 157, CG: 183
Init-CG: 158, CG: 182
Init-CG: 155, CG: 183
Init-CG: 154, CG: 183
```
"""
function eigcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, nvec::Int, spdim::Int)
  r, Ap, p, res_norm = similar(x), similar(x), similar(x), similar(x)
  #
  n = size(x)[1]
  V = Array{T}(undef, (n, spdim))
  VtAV = zeros(T, spdim, spdim)
  Y = zeros(T, (spdim, 2 * nvec))
  tvec = Vector{T}(undef, n)
  just_restarted = false
  #
  it = 1
  r = b - A * x
  rTr = dot(r, r)
  p .= r
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
    if ivec == spdim
      tvec -= beta * Ap
    end
    axpby!(1, r, beta, p) # p = beta * p + r
    it += 1
    res_norm[it] = sqrt(rTr)
    #
    VtAV[ivec, ivec] += 1 / alpha
    if just_restarted
      tvec += Ap
      nev = ivec - 1
      VtAV[1:nev, ivec] = V[:, 1:nev]' * (tvec / res_norm[it - 1])
      just_restarted = false
    end
    #
    if ivec == spdim
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
      tvec = - beta * Ap
      just_restarted = true
      #VtAV[1:nev, ivec] = V[:, 1:nev]' * (A * V[:, ivec]) # Matrix-vector product avoided with tvec
    else
      ivec += 1
      V[:, ivec] = r / res_norm[it]
      VtAV[ivec - 1, ivec] = - sqrt(beta) / alpha
      VtAV[ivec, ivec] = beta / alpha
    end
  end
  return x, it, res_norm[1:it], V[:, 1:nvec]
end

function eigpcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, M, nvec::Int, spdim::Int)
  nothing
end
