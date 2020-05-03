const eps = 1e-7

function pcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, precond::BjPrecond)
  r, z, Ap, res_norm = similar(x), similar(x), similar(x), similar(x)
  it = 1
  r[1:end] = b - A * x
  rTr = r'r
  z[1:end] = invM(r, precond)::Vector{T}
  rTz = r'z
  p = copy(z)
  res_norm[it] = sqrt(rTr)
  bnorm = sqrt(b'b)
  tol = eps * bnorm
  while (it < A.n) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = p'Ap
    alpha = rTz / d
    beta = 1. / rTz
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = r'r
    z[1:end] = invM(r, precond)::Vector{T}
    rTz = r'z
    beta *= rTz
    axpby!(1, z, beta, p) # p = beta * p + z
    it += 1
    res_norm[it] = sqrt(rTr)
  end
  return x, it, res_norm[1:it]
end
