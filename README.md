## MyRecycledKrylovSolvers.jl and MyPreconditioners.jl

##### Julia code of conjugate gradient (CG) algorithms with recycled Krylov subspaces for use in solves of sequences of linear systems with multiple sparse symmetric positive definite (SPD) matrices and/or right-hand sides.



Author: Nicolas Venkovic.

email: venkovic@gmail.com.



#### Dependencies:

 - Julia 1.4, Python 3.6.
 - Python packages: SciPy 1.4.1, PyAmg 4.0.0. 
 - Julia packages: LinearAlgebra.jl, PyCall.jl, SparseArrays,.jl, SuiteSparse.jl. 



#### Running examples: 

__Example 1: Multiple right-hand sides (one constant sparse SPD matrix)__

```bash
julia Example01_mrhs.jl
python Example01_mrhs.py
```

This example solves the linear systems <img src="/tex/312d8203f3f175f1b04eeddb5f0795b6.svg?invert_in_darkmode&sanitize=true" align=middle width=63.926867399999985pt height=22.831056599999986pt/> defined by `nsmp`=`10` samples of identically independently distributed (i.i.d.) <img src="/tex/14ec36abab994638a29863f0f4526ec7.svg?invert_in_darkmode&sanitize=true" align=middle width=13.259167349999991pt height=22.831056599999986pt/>. The `n`-by-`n` matrix <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> is the square of a random SPD tridiagonal matrix. We set `n`=`1_000_000` and use eigPCG (Stathopoulos and Orginos, 2010) with block Jacobi (bJ) preconditioners using different numbers (10, 20, 30) of diagonal blocks. The `nvec` least dominant (LD) eigenvector approximations of <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> extracted from the solve of the first system <img src="/tex/87aede2bbf21e226d13daf11f9cef89c.svg?invert_in_darkmode&sanitize=true" align=middle width=64.62322019999998pt height=22.831056599999986pt/>, which we store by columns in a matrix <img src="/tex/6bac6ec50c01592407695ef84f457232.svg?invert_in_darkmode&sanitize=true" align=middle width=13.01596064999999pt height=22.465723500000017pt/>, are not  accurate enough. That is to say, generating an initial iterate for the second solve whose residual is orthogonal to the range of <img src="/tex/6bac6ec50c01592407695ef84f457232.svg?invert_in_darkmode&sanitize=true" align=middle width=13.01596064999999pt height=22.465723500000017pt/>, e.g., <img src="/tex/d5e735f01d78c97f769f343e19961fd2.svg?invert_in_darkmode&sanitize=true" align=middle width=128.79679889999997pt height=27.6567522pt/>, does not significantly accelerate the iterative solve. Thus, eigPCG is used incrementally (Stathopoulos and Orginos, 2010) for `nincr`=`3` solves. After every such solve, `nvec` additional (column) eigenvector approximations are appended to <img src="/tex/6bac6ec50c01592407695ef84f457232.svg?invert_in_darkmode&sanitize=true" align=middle width=13.01596064999999pt height=22.465723500000017pt/>. Meanwhile, every initial iterate is set such that its residual is orthogonal to the range of the incrementally growing <img src="/tex/6bac6ec50c01592407695ef84f457232.svg?invert_in_darkmode&sanitize=true" align=middle width=13.01596064999999pt height=22.465723500000017pt/>. Once `nincr` systems have been solved by eigPCG, <img src="/tex/6bac6ec50c01592407695ef84f457232.svg?invert_in_darkmode&sanitize=true" align=middle width=13.01596064999999pt height=22.465723500000017pt/> remains constant as it is used to generate the initial iterates of all the next systems solved by Init-PCG (Erhel & Guyomarc'h, 2000). The black curves in the figure below are the convergence histories of the first systems in the sequence. The convergence histories are made gradually more colorful throughout the sampled sequence. Note that the incremental eigPCG procedure enables a near 90% decrease of the number of required solver iterations when using a bJ preconditioner with 30 diagonal blocks (i.e., bJ30). The relative acceleration obtained is less significant when using less blocks.

![](./Example1_mrhs.png)

This example works properly. However, this approach can be pushed to its limit by increasing the number of increments `nincr`. As the total number `nincr` * `nvec` of approximate eigenvectors increases, <img src="/tex/6bac6ec50c01592407695ef84f457232.svg?invert_in_darkmode&sanitize=true" align=middle width=13.01596064999999pt height=22.465723500000017pt/> tends to lose rank, which makes the computation of an initial iterate with an orthogonal residual more difficult. To alleviate this effect, one can orthogonalize the vectors in <img src="/tex/6bac6ec50c01592407695ef84f457232.svg?invert_in_darkmode&sanitize=true" align=middle width=13.01596064999999pt height=22.465723500000017pt/>, at a computational cost O((`nincr` * `nvec`)^2 * `n`). See Stathopoulos and Orginos (2010) for further details.



__Example 2: Multiple correlated sparse SPD matrices (one constant right-hand side)__

```bash
julia Example02_mops.jl
python Example02_mops.py
```

This example solves the linear systems <img src="/tex/ec91cca0d4d5b89ffd6ffceccee0139b.svg?invert_in_darkmode&sanitize=true" align=middle width=64.74878025pt height=22.831056599999986pt/> defined by `nsmp`=`10` samples <img src="/tex/fde01d7c410d7e41084b8833273ceae7.svg?invert_in_darkmode&sanitize=true" align=middle width=18.53316959999999pt height=22.465723500000017pt/> of a random walk. The `n`-by-`n` matrix <img src="/tex/c74f257c1a844c30acb274ac45ecd397.svg?invert_in_darkmode&sanitize=true" align=middle width=18.881345999999994pt height=22.465723500000017pt/> is the square of a random SPD tridiagonal matrix. We set `n`=`1_000_000` and use eigPCG (Stathopoulos and Orginos, 2010) with block Jacobi (bJ) preconditioners using different numbers (10, 20, 30) of diagonal blocks. The `nvec` LD eigenvector approximations of <img src="/tex/c74f257c1a844c30acb274ac45ecd397.svg?invert_in_darkmode&sanitize=true" align=middle width=18.881345999999994pt height=22.465723500000017pt/> extracted from the eigPCG solve of <img src="/tex/6d5e99d40bf83b73f6b593b6454096b1.svg?invert_in_darkmode&sanitize=true" align=middle width=65.44513304999998pt height=22.831056599999986pt/> are stored by columns in a matrix <img src="/tex/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=22.465723500000017pt/>. The range of <img src="/tex/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=22.465723500000017pt/> is then used as a deflation subspace for the iterative eigDef-PCG solve of <img src="/tex/f132281d0f1d90387f88647682e195b0.svg?invert_in_darkmode&sanitize=true" align=middle width=65.44513304999998pt height=22.831056599999986pt/> during which eigenvector approximations of <img src="/tex/0a3132987975418a383f22eef58769cb.svg?invert_in_darkmode&sanitize=true" align=middle width=18.881345999999994pt height=22.465723500000017pt/> are extracted in a similar as in Stathopoulos and Orginos, (2010). These approximate eigenvectors are used to update <img src="/tex/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=22.465723500000017pt/> before the next eigDef-PCG solve. The eigDef-PCG algorithm is referred to as RR-LO-TR-eigDef-PCG in Venkovic et al. (2020). The black curves in the figure below are the convergence histories of the first systems in the sequence. The convergence histories are made gradually more colorful throughout the sampled sequence. Note that the incremental eigPCG procedure enables a near 75% decrease of the number of required solver iterations when using a bJ preconditioner with 30 diagonal blocks (i.e., bJ30). The relative acceleration obtained is less significant when using less blocks.

![](./Example2_mops.png)

This example works properly. However, this approach can be pushed to its limit by (i) scaling the matrix increment of the random walk used to compute the matrices <img src="/tex/fde01d7c410d7e41084b8833273ceae7.svg?invert_in_darkmode&sanitize=true" align=middle width=18.53316959999999pt height=22.465723500000017pt/>, (ii) decreasing the threshold of convergence on the backward error, (iii) increasing `nvec` or `spdim`, or (iv) increasing the dimension `n` of the problem. When doing so, the iterated residual <img src="/tex/212f899c5235a861a1f6146dc8d1582f.svg?invert_in_darkmode&sanitize=true" align=middle width=13.520829299999992pt height=14.15524440000002pt/> tends to lose it orthogonality with respect to the deflation subspace, in which case eigDef-PCG tends to not converge, and even becomes unstable. This problem, which was described in Saad et al. (1999), can be alleviated by setting <img src="/tex/23d043c17058f8cf3d78a5c07d0765ab.svg?invert_in_darkmode&sanitize=true" align=middle width=211.15942979999997pt height=27.6567522pt/> at the end of each solver iteration. This modification, which entails a computational cost O(`nvec` * `n`) at each iteration, does help, but does not always solve the problem.



#### Functions of MyRecycledKrylovSolvers.jl:

The default type `T`=`Float64` can be changed in MyRecycledKrylovSolvers.jl, in which case it should also be changed in MyPreconditioners.jl. Every function described underneath contains `jldoctest` scripting examples.

__cg.jl__:

- `cg` (`A`::SparseMatrixCSC{`T`}, `b`::Vector{`T`}, `x`::Vector{`T`}) 

  Computes iterates of CG (Saad, 2003).

  Inputs:

  `A`::SparseMatrixCSC{`T`}. Sparse CSC SPD `n`-by-`n` matrix. 

  `b`::Vector{`T`}. Right-hand side. 

  `x`::Vector{`T`}. Initial iterate. 

  Returns: `x`::Vector{`T`}, `it`::`Int`, `res_norm`::Vector{`T`}. 

  `x`::Vector{`T`}. Iterate at termination.

  `it`::`Int`. Number of iterations completed at termination.

  `res_norm`::Vector{`T`}. Norm of every iterated residual prior to termination. 

- `pcg` (`A`::SparseMatrixCSC{`T`}, `b`::Vector{`T`}, `x`::Vector{`T`}, `M`)

  Computes iterates of PCG (Saad, 2003).

  Inputs:

  `M`. SPD preconditioner. May be custom typed, or not. Must support the operation `M` \ `b` and return a `n`-dimensional Vector{`T`} as a means to apply the inverse preconditioner. 

  Returns: `x`::Vector{`T`}, `it`::`Int`, `res_norm`::Vector{`T`}. 

  

__initcg.jl__:

- `initcg` (`A`::SparseMatrixCSC{`T`}, `b`::Vector{`T`}, `x`::Vector{`T`}, `W`::Array{`T`,`2`}) 

  Computes iterates of Init-CG (Erhel & Guyomarc'h, 2000). Used to solve <img src="/tex/70681e99f542745bf6a0c56bd4600b39.svg?invert_in_darkmode&sanitize=true" align=middle width=50.69621369999999pt height=22.831056599999986pt/> with an SPD matrix <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> when a set of linearly independent vectors <img src="/tex/4b4518f1b7f0fb1347fa21506ebafb19.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, <img src="/tex/f7eb0e840408d84a0c156d6efb611f3e.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, ... is known and such that Span{<img src="/tex/4b4518f1b7f0fb1347fa21506ebafb19.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, <img src="/tex/f7eb0e840408d84a0c156d6efb611f3e.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, ...} is "approximately" invariant under the action of <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/>. Then an initial guess may be generated which is deflated of the solution projected onto the invariant subspace. Initializing a regular CG solve with such a deflated initial guess can result in improvements of the convergence behavior.

  Inputs:

  `A`::SparseMatrixCSC{`T`}. Sparse CSC SPD `n`-by-`n` matrix. 

  `b`::Vector{`T`}. Right-hand side. 

  `x`::Vector{`T`}. Initial iterate. 

  `W`::Array{`T`,`2`}. Matrix of `nvec` linearly independent `n`-dimensional vectors. 

  Returns: `x`::Vector{`T`}, `it`::`Int`, `res_norm`::Vector{`T`}. 

  `x`::Vector{`T`}. Iterate at termination.

  `it`::`Int`. Number of iterations completed at termination.

  `res_norm`::Vector{`T`}. Norm of every iterated residual prior to termination. 

- `initpcg` (`A`::SparseMatrixCSC{`T`}, `b`::Vector{`T`}, `x`::Vector{`T`}, `W`::Array{`T`,`2`}) 

  Computes iterates of Init-PCG (Erhel & Guyomarc'h, 2000). Used to solve <img src="/tex/70681e99f542745bf6a0c56bd4600b39.svg?invert_in_darkmode&sanitize=true" align=middle width=50.69621369999999pt height=22.831056599999986pt/> with an SPD matrix <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> and an SPD preconditioner <img src="/tex/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73973739999999pt height=22.465723500000017pt/> when a set of linearly independent vectors <img src="/tex/4b4518f1b7f0fb1347fa21506ebafb19.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, <img src="/tex/f7eb0e840408d84a0c156d6efb611f3e.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, ... is known and such that Span{<img src="/tex/4b4518f1b7f0fb1347fa21506ebafb19.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, <img src="/tex/f7eb0e840408d84a0c156d6efb611f3e.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, ...} is "approximately" invariant under the action of <img src="/tex/15ec5ccdcd83c454ab399827e8d32e2a.svg?invert_in_darkmode&sanitize=true" align=middle width=47.716972049999995pt height=26.76175259999998pt/>. Then an initial guess may be generated which is deflated of the solution projected onto the invariant subspace. Initializing a regular PCG solve with such a deflated initial guess can result in improvements of the convergence behavior.

  Inputs:

  `M`. SPD preconditioner. May be custom typed, or not. Must support the operation `M` \ `b` and return a `n`-dimensional Vector{`T`} as a means to apply the inverse preconditioner. 

  Returns: `x`::Vector{`T`}, `it`::`Int`, `res_norm`::Vector{`T`}. 

  

__defcg.jl__:

- `defcg` (`A`::SparseMatrixCSC{`T`}, `b`::Vector{`T`}, `x`::Vector{`T`}, `W`::Array{`T`,`2`}) 

  Computes iterates of Def.-CG (Saad et al., 2000). Used to solve <img src="/tex/70681e99f542745bf6a0c56bd4600b39.svg?invert_in_darkmode&sanitize=true" align=middle width=50.69621369999999pt height=22.831056599999986pt/> with an SPD matrix <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> when a set of linearly independent vectors <img src="/tex/4b4518f1b7f0fb1347fa21506ebafb19.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, <img src="/tex/f7eb0e840408d84a0c156d6efb611f3e.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, ... is known such that Span{<img src="/tex/4b4518f1b7f0fb1347fa21506ebafb19.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, <img src="/tex/f7eb0e840408d84a0c156d6efb611f3e.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, ...} is "approximately" invariant under the action of <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/>. The sequence of iterates of Def-CG is equivalent to a post-processed sequence of the regular CG solve of a deflated version of the linear system, with guaranteed decrease of the condition number. Remark: if Span{<img src="/tex/4b4518f1b7f0fb1347fa21506ebafb19.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, <img src="/tex/f7eb0e840408d84a0c156d6efb611f3e.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, ...} is exactly invariant under the action of <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/>, one should use Init-CG instead of Def-CG because both algorithms would then have equally positive impacts on convergence, but Def-CG requires an additional computational cost at every solver iteration.

  Inputs:

  `A`::SparseMatrixCSC{`T`}. Sparse CSC SPD `n`-by-`n` matrix. 

  `b`::Vector{`T`}. Right-hand side. 

  `x`::Vector{`T`}. Initial iterate. 

  `W`::Array{`T`,`2`}. Matrix of `nvec` linearly independent `n`-dimensional vectors.

  Returns: `x`::Vector{`T`}, `it`::`Int`, `res_norm`::Vector{`T`}. 

  `x`::Vector{`T`}. Iterate at termination.

  `it`::`Int`. Number of iterations completed at termination.

  `res_norm`::Vector{`T`}. Norm of every iterated residual prior to termination. 

- `eigdefcg` (`A`::SparseMatrixCSC{`T`}, `b`::Vector{`T`}, `x`::Vector{`T`}, `W`::Array{`T`,`2`},  `spdim`::`Int`) 

  Computes iterates of RR-LO-TR-Def-CG (Venkovic et al., 2020), here referred to as eigDef-CG. Works as a combination of eigCG and Def-CG. The linear solve is deflated as in Def-CG, and approximate least dominant eigenvectors of <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> are computed throughout the solve in a similar way as in eigCG. This algorithm is an alternative to the incremental eigCG algorithm when solving for a sequence of systems <img src="/tex/8b52d1ced847605a2cc0351e3cc6c22a.svg?invert_in_darkmode&sanitize=true" align=middle width=63.926867399999985pt height=22.831056599999986pt/> with a constant SPD matrix <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> and different right-hand sides <img src="/tex/14ec36abab994638a29863f0f4526ec7.svg?invert_in_darkmode&sanitize=true" align=middle width=13.259167349999991pt height=22.831056599999986pt/>. This algorithm should be the method of choice when solving a sequence of linear systems of the form <img src="/tex/f2c8905efdb3042fa8c7d8d7df49bb85.svg?invert_in_darkmode&sanitize=true" align=middle width=70.95315149999999pt height=22.831056599999986pt/> with correlated SPD matrices <img src="/tex/c74f257c1a844c30acb274ac45ecd397.svg?invert_in_darkmode&sanitize=true" align=middle width=18.881345999999994pt height=22.465723500000017pt/>, <img src="/tex/0a3132987975418a383f22eef58769cb.svg?invert_in_darkmode&sanitize=true" align=middle width=18.881345999999994pt height=22.465723500000017pt/>, ... Examples are shown in the `jldoctest` for each type of problem.

  Inputs:

  `spdim`::`Int`. Maximum dimension of the eigen-search space. Must be such that `spdim` > `2` * `nvec`.

  Returns: `x`::Vector{`T`}, `it`::`Int`, `res_norm`::Vector{`T`}, `W`::Array{`T`,`2`}.

  `W`::Array{`T`,`2`}. `nvec`approximate LD column eigenvectors of `A`.

- `defpcg` (`A`::SparseMatrixCSC{`T`}, `b`::Vector{`T`}, `x`::Vector{`T`}, `W`::Array{`T`,`2`})

  Computes iterates of Def.-PCG (Saad et al., 2000). Used to solve <img src="/tex/66a8a0c17c80a313cb880fcc6d6392f3.svg?invert_in_darkmode&sanitize=true" align=middle width=50.69621369999999pt height=22.831056599999986pt/> with an SPD matrix <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> and an SPD preconditioner <img src="/tex/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73973739999999pt height=22.465723500000017pt/>, when a set of linearly independent vectors <img src="/tex/4b4518f1b7f0fb1347fa21506ebafb19.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, <img src="/tex/f7eb0e840408d84a0c156d6efb611f3e.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, ... is known such that Span{<img src="/tex/4b4518f1b7f0fb1347fa21506ebafb19.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, <img src="/tex/f7eb0e840408d84a0c156d6efb611f3e.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, ...} is "approximately" invariant under the action of <img src="/tex/15ec5ccdcd83c454ab399827e8d32e2a.svg?invert_in_darkmode&sanitize=true" align=middle width=47.716972049999995pt height=26.76175259999998pt/>. The sequence of iterates of Def-PCG is equivalent to a post-processed sequence of the regular CG solve of a deflated and split-preconditioned version of the linear system, with guaranteed decrease of the condition number. Remark: if Span{<img src="/tex/4b4518f1b7f0fb1347fa21506ebafb19.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, <img src="/tex/f7eb0e840408d84a0c156d6efb611f3e.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, ...} is exactly invariant under the action of <img src="/tex/15ec5ccdcd83c454ab399827e8d32e2a.svg?invert_in_darkmode&sanitize=true" align=middle width=47.716972049999995pt height=26.76175259999998pt/>, one should use Init-PCG instead of Def-PCG because both algorithms would then have equally positive impacts on convergence, but Def-PCG requires an additional computational cost at every solver iteration.

  Inputs:

  `M`. SPD preconditioner. May be custom typed, or not. Must support the operation `M` \ `b` and return a `n`-dimensional Vector{`T`} as a means to apply the inverse preconditioner. 

  Returns: `x`::Vector{`T`}, `it`::`Int`, `res_norm`::Vector{`T`}.

- `eigdefpcg` (`A`::SparseMatrixCSC{`T`}, `b`::Vector{`T`}, `x`::Vector{`T`}, `M`, `W`::Array{`T`,`2`},  `spdim`::`Int`) 

  Computes iterates of RR-LO-TR-Def-PCG (Venkovic et al., 2020), here referred to as eigDef-PCG. Works as a combination of eigPCG and Def-PCG. The linear solve is deflated as in Def-PCG, and approximate least dominant right eigenvectors of <img src="/tex/15ec5ccdcd83c454ab399827e8d32e2a.svg?invert_in_darkmode&sanitize=true" align=middle width=47.716972049999995pt height=26.76175259999998pt/> are computed throughout the solve in a similar way as in eigPCG. This algorithm is an alternative to the incremental eigPCG algorithm when solving for a sequence of systems <img src="/tex/8b52d1ced847605a2cc0351e3cc6c22a.svg?invert_in_darkmode&sanitize=true" align=middle width=63.926867399999985pt height=22.831056599999986pt/> with constant SPD <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> and <img src="/tex/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73973739999999pt height=22.465723500000017pt/>, and different right-hand sides <img src="/tex/14ec36abab994638a29863f0f4526ec7.svg?invert_in_darkmode&sanitize=true" align=middle width=13.259167349999991pt height=22.831056599999986pt/>. This algorithm should be the method of choice when solving a sequence of linear systems of the form <img src="/tex/f2c8905efdb3042fa8c7d8d7df49bb85.svg?invert_in_darkmode&sanitize=true" align=middle width=70.95315149999999pt height=22.831056599999986pt/> with correlated SPD matrices <img src="/tex/c74f257c1a844c30acb274ac45ecd397.svg?invert_in_darkmode&sanitize=true" align=middle width=18.881345999999994pt height=22.465723500000017pt/>, <img src="/tex/0a3132987975418a383f22eef58769cb.svg?invert_in_darkmode&sanitize=true" align=middle width=18.881345999999994pt height=22.465723500000017pt/>, ... Examples are shown in the `jldoctest` for each type of problem.

  Returns: `x`::Vector{`T`}, `it`::`Int`, `res_norm`::Vector{`T`}, `W`::Array{`T`,`2`}.

  `W`::Array{`T`,`2`}. `nvec`approximate LD column right eigenvectors of `M`^`-1` * `A`.

  

__eigcg.jl__:

- `eigcg` (`A`::SparseMatrixCSC{`T`}, `b`::Vector{`T`}, `x`::Vector{`T`}, `nvec`::`Int`, `spdim`::`Int`) 

  Computes iterates of eigCG (Stathopoulos & Orginos, 2010). Used at the beginning of a solving procedure of linear systems <img src="/tex/8b52d1ced847605a2cc0351e3cc6c22a.svg?invert_in_darkmode&sanitize=true" align=middle width=63.926867399999985pt height=22.831056599999986pt/> with a constant SPD matrix <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/>, and different right-hand sides <img src="/tex/14ec36abab994638a29863f0f4526ec7.svg?invert_in_darkmode&sanitize=true" align=middle width=13.259167349999991pt height=22.831056599999986pt/>. eigCG may be run once (or incrementally) to generate approximate least dominant eigenvectors of <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/>. These approximate eigenvectors are then used to generate a deflated initial guess with the Init-CG algorithm. Incremental eigCG should be used when the solve of the first system ends before accurate eigenvector approximations can be obtained by eigCG, which then limits the potential speed-up obtained for the subsequent Init-CG solves. See Example for typical use and implementation of the Incremental eigCG algorithm (Stathopoulos & Orginos, 2010).

  Inputs:

  `A`::SparseMatrixCSC{`T`}. Sparse CSC SPD `n`-by-`n` matrix. 

  `b`::Vector{`T`}. Right-hand side. 

  `x`::Vector{`T`}. Initial iterate. 

  `nvec`::`Int`. Number of wanted approximate eigenvectors. Typically, `nvec` << `n`.

  `spdim`::`Int`. Maximum dimension of the eigen-search space. Must be such that `spdim` > `2` * `nvec`.

  Returns: `x`::Vector{`T`}, `it`::`Int`, `res_norm`::Vector{`T`}, `W`::Array{`T`,`2`}.

  `x`::Vector{`T`}. Iterate at termination.

  `it`::`Int`. Number of iterations completed at termination.

  `res_norm`::Vector{`T`}. Norm of every iterated residual prior to termination. 

  `W`::Array{`T`,`2`}. `nvec`approximate LD column eigenvectors of `A`.

- `eigpcg` (`A`::SparseMatrixCSC{`T`}, `b`::Vector{`T`}, `x`::Vector{`T`}, `M`, `nvec`::`Int`, `spdim`::`Int`) 

  Computes iterates of eigPCG (Stathopoulos & Orginos, 2010). Used at the beginning of a solving procedure of linear systems A xs = bs with
constant SPD matrix A and SPD preconditioner M, and different right-hand sides <img src="/tex/14ec36abab994638a29863f0f4526ec7.svg?invert_in_darkmode&sanitize=true" align=middle width=13.259167349999991pt height=22.831056599999986pt/>. eigPCG may be run once (or incrementally) to generate approximate least dominant right eigenvectors of <img src="/tex/15ec5ccdcd83c454ab399827e8d32e2a.svg?invert_in_darkmode&sanitize=true" align=middle width=47.716972049999995pt height=26.76175259999998pt/>. These approximate eigenvectors are then used to generate a deflated initial guess with the Init-PCG algorithm. Incremental eigPCG should be used when the solve of the first system ends before accurate eigenvector approximations can be obtained by eigPCG, which then limits the potential speed-up obtained for the subsequent Init-PCG solves. See Examples for typical use and implementation of the Incremental eigPCG algorithm (Stathopoulos & Orginos, 2010).
  
Returns:
  
`W`::Array{`T`,`2`}. `nvec`approximate LD column right eigenvectors of `M`^`-1` * `A`.
  
  

#### Functions of MyPreconditioners.jl:

The default type `T`=`Float64` can be changed in MyPreconditioners.jl.

- `BJPreconditioner` (`nb`::`Int`, `A`::SparseMatrixCSC{`T`}) 

  Prepares a block Jacobi preconditioner with `nb` diagonal blocks.

  Returns: `M`::BJop.

  `M`::BJop. BJ preconditioner. Supports the operation `M` \ `b` and return a `n`-dimensional Vector{`T`} as a means to apply the inverse preconditioner. 

- `Chol32Preconditioner` (`A`::SparseMatrixCSC{`T`}) 

  Prepares a Cholesky factorization of `Float32`(`A`).

  Returns: `M`::SuiteSparse.CHOLMOD.Factor{`T`}.

  `M`::SuiteSparse.CHOLMOD.Factor. Cholesky preconditioner of `Float32`(`A`). Supports the operation `M` \ `b` and return a `n`-dimensional Vector{`T`} as a means to apply the inverse preconditioner. 

- `Chol16Preconditioner` (`A`::SparseMatrixCSC{`T`}) 

  Prepares a Cholesky factorization of `Float16`(`A`).

  Returns: `M`::PyObject <`n`x`n` _CustomLinearOperator with dtype=float64>.

  `M`::SuiteSparse.CHOLMOD.Factor. Cholesky preconditioner of `Float16`(`A`). Supports the operation `M` \ `b` and return a `n`-dimensional Vector{`T`} as a means to apply the inverse preconditioner. 

- `AMGPreconditioner` (`A`::SparseMatrixCSC{`T`}) 

  Prepares a single V-cycle of a smoothed aggregation AMG solver of `A` as preconditioner.

  Returns: `M`::PyObject <`n`x`n` _CustomLinearOperator with dtype=float64>.

  `M`::PyObject. AMG preconditioner of `A`. Supports the operation `M` \ `b` and return a `n`-dimensional Vector{`T`} as a means to apply the inverse preconditioner. 
  
  


#### Known issues:

 -  The AMG preconditioner does not accelerate solver convergence as expected. Note that, prior to constructing the AMG preconditioner using PyAmg, the reference Julia CSC sparse array is properly converted into a SciPy CSR sparse array. We currently do not know why this wrapped preconditioner under-performs compared to how we know it should do.

   

#### References:

 -  __Erhel, J. & Guyomarc'h, F.__ (2000) **An augmented conjugate gradient method for solving consecutive symmetric positive definite linear systems**, SIAM Journal on Matrix Analysis and Applications, SIAM, 21, 1279-1299.
 -  __Giraud, L.; Ruiz, D. & Touhami, A.__ (2006) **A comparative study of iterative solvers exploiting spectral information for SPD systems**, SIAM Journal on Scientific Computing, SIAM, 27, 1760-1786.
 -  __Saad, Y.__ (2003) **Iterative methods for sparse linear systems**, SIAM, 82.
 -  __Saad, Y.; Yeung, M.; Erhel, J. & Guyomarc'h, F.__ (1999) **Deflated Version of the Conjugate Gradient Algorithm**, SIAM Journal on Scientific Computing, SIAM, 21, 1909-1926.
 -  __Stathopoulos, A. & Orginos, K.__ (2010) **Computing and deflating eigenvalues while solving multiple right-hand side linear systems with an application to quantum chromodynamics**, SIAM Journal on Scientific Computing, SIAM, 32, 439-462.
 -  __Venkovic, N.; Mycek, P.; Giraud, L.; Le Maître, O.__ (submitted in 2020) **Recycling Krylov subspace strategies for sequences of sampled stochastic elliptic equations**, SIAM Journal on Scientific Computing, SIAM, under review.