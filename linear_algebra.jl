using LinearAlgebra
using SparseArrays
using IterativeSolvers

function get_spd_sparse(n)
  A = Bidiagonal(rand(n), rand(n - 1), :U)
  A += A'
  return A'A
end

function get_spd_sparse(n, rel_nnz)
  A = Bidiagonal(rand(n), rand(n - 1), :U)
  A += A'
  dA = sprand(n, n, rel_nnz)
  dA = (dA + dA')/2
  A += dA
  return A'A
end;


n = 5000
rel_nnz = .2

@time a = get_spd_sparse(n)
@time res_a = lobpcg(a, true, 5)

@time b = get_spd_sparse(n, rel_nnz)
@time res_b = lobpcg(b, true, 5)


"b = rand(Float64, n)"



