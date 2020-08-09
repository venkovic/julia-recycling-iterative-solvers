import numpy as np
import scipy.sparse as sp
from pyamg import smoothed_aggregation_solver
import os

def load_CSC_sparse_array_from_jl(fname, n):
  colptr = np.load(fname + "-" + str(n) + "_colptr.npy") - 1
  nzval = np.load(fname + "-" + str(n) + "_nzval.npy")
  rowval = np.load(fname + "-" + str(n) + "_rowval.npy") - 1
  A = sp.csc_matrix((nzval, rowval, colptr), (n, n))
  A = sp.csr_matrix(A)
  return A

fname = "random-penta"
n = 1000

os.system("julia _TestAMG.jl")

A = load_CSC_sparse_array_from_jl(fname, n)
b = np.load("b.npy")

ml = smoothed_aggregation_solver(A)
M = ml.aspreconditioner(cycle='V')

x = np.load(fname + "-" + str(n) + "_x.npy")
res = np.linalg.norm(x - M(b))
tol = 1e-7

if res < tol:
  print("PyAMG runs properly when called from Julia.")
else:
  print("PyAMG does not run properly when called from Julia.")

os.system("rm %s-%d_colptr.npy" % (fname, n))
os.system("rm %s-%d_nzval.npy" % (fname, n))
os.system("rm %s-%d_rowval.npy" % (fname, n))
os.system("rm b.npy")
os.system("rm %s-%d_x.npy" % (fname, n))
