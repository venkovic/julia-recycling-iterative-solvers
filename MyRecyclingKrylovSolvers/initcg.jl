"""
initcg(A, b, x, W)

Performs Init-CG (Erhel & Guyomarc'h, 2000).

Used to solve A x = b with an SPD matrix A when a set of linearly independent
vectors w1, w2, ... is known and such that Span{w1, w2, ...} is (at least
"approximately") invariant under the action of A. Then an initial guess may be
generated which is deflated of the solution projected onto the invariant
subspace. Initializing a regular CG solve with such a deflated initial guess can
result in improvements on the convergence behavior.

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
julia>
using LinearAlgebra: SymTridiagonal;
using SparseArrays: sparse, SparseMatrixCSC;
using Arpack: eigs;
push!(LOAD_PATH, "./MyRecyclingKrylovSolvers");
using MyRecyclingKrylovSolvers: initcg, defcg, cg;
using Random: seed!
seed!(1234);
const n = 1_000;
const T = Float64;
A = sparse(SymTridiagonal(2 .+ .5 * rand(T, n), -1 .+ .05 * rand(T, n-1)));
A = A * A;
function mrhs_initcg(A::SparseMatrixCSC{T}, nvec::Int, nsmp::Int)
  _, W = eigs(A; nev=nvec, which=:SM);
  println("\\n* Init-CG results *");
  for ismp in 1:nsmp
    b = rand(T, n);
    _, itinitcg, _ = initcg(A, b, zeros(T, n), W);
    _, itdefcg, _ = defcg(A, b, zeros(T, n), W);
    _, itcg, _ = cg(A, b, zeros(T, n));
    println("Init-CG: ", itinitcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
  end
end
nsmp, nvec = 5, 20;
mrhs_initcg(A, nvec, nsmp);

* Init-CG results *
Init-CG: 140, Def-CG: 140, CG: 183
Init-CG: 141, Def-CG: 141, CG: 183
Init-CG: 141, Def-CG: 141, CG: 183
Init-CG: 140, Def-CG: 140, CG: 184
Init-CG: 140, Def-CG: 140, CG: 183
```
"""
function initcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, W::Array{T,2})
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
  p .= r
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

function initpcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, M, W::Array{T,2})
  nothing
end
