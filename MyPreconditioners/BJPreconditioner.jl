struct op
  n::Int
  nb::Int
  bsize::Int
  chol::Vector{Factor{T}}
end

function slice(i::Int, nb::Int, bsize::Int, n::Int)
  if i < nb
    return ((i - 1 ) * bsize + 1):(i * bsize)
  else
    return ((i - 1 ) * bsize + 1):n
  end
end

function BJPreconditioner(nb::Int, A::SparseMatrixCSC{T})
  n = A.n
  bsize = Int(floor((n // nb)))
  chol = map(i -> cholesky(A[slice(i, nb, bsize, n), slice(i, nb, bsize, n)]), 1:nb)
  return op(n, nb, bsize, chol)
end

function invM(x::Vector{T}, precond::op)
  y = similar(x)
  for i in 1:precond.nb
    y[slice(i, precond.nb, precond.bsize, precond.n)] = precond.chol[i] \ x[slice(i, precond.nb, precond.bsize, precond.n)]
  end
  return y
end

import Base: \
(\)(M::op, x::Vector{T}) = invM(x::Vector{T}, M::op)
