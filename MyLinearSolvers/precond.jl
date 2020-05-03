struct BjPrecond
  n::Int
  nb::Int
  bsize::Int
  chol::Vector{Factor{T}}
end

function slice(i::Int, nb::Int, bsize::Int, n::Int)
  if i < nb
    ((i - 1 ) * bsize + 1):(i * bsize)
  else
    ((i - 1 ) * bsize + 1):n
  end
end

function set_precond(nb::Int, A::SparseMatrixCSC{T})
  n = A.n
  bsize = Int(floor((n // nb)))
  chol = map(i -> cholesky(A[slice(i, nb, bsize, n), slice(i, nb, bsize, n)]), 1:nb)
  return BjPrecond(n, nb, bsize, chol)
end

function invM(x::Vector{T}, precond::BjPrecond)
  y = similar(x)
  for i in 1:precond.nb
    y[slice(i, precond.nb, precond.bsize, precond.n)] = precond.chol[i] \ x[slice(i, precond.nb, precond.bsize, precond.n)]
  end
  return y
end

struct AmgPrecond
  nothing
end
