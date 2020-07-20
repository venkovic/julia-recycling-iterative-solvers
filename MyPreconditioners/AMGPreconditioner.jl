function AMGPreconditioner(A::SparseMatrixCSC{T})
  indptr = A.colptr .- 1
  indices = A.rowval .- 1
  vals = copy(A.nzval)
  B = pysparse.csc_matrix((vals, indices, indptr), shape=(A.n, A.m)).tocsr()
  ml = pyamg.smoothed_aggregation_solver(B)
  amg_op = ml.aspreconditioner(cycle="V")
  return amg_op
end

import Base: \
(\)(amg_op::PyObject, x::Vector{T}) = amg_op(x::Vector{T})
