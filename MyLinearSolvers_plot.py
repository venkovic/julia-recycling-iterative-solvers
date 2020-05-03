import numpy as np
import pylab as pl

pl.rcParams['text.usetex'] = True
params = {'text.latex.preamble' : [r'\usepackage{amssymb}',
                                   r'\usepackage{amsmath}']}
pl.rcParams.update(params)
pl.rcParams['axes.labelsize'] = 20
pl.rcParams['axes.titlesize'] = 20
pl.rcParams['legend.fontsize'] = 20
pl.rcParams['xtick.labelsize'] = 20
pl.rcParams['ytick.labelsize'] = 20
pl.rcParams['legend.numpoints'] = 1

nsmp = 100
nblocks = (4, 8, 12, 16)
cols = ('r', 'b', 'g', 'y')

fig, ax = pl.subplots()
for ibJ, nblock in enumerate(nblocks):
  its = np.load('bJ%d-pcg.its.npy' % nblock)
  res_norms = np.load('bJ%d-pcg.res_norms.npy' % nblock)
  sum_it = 0
  for ismp in range(nsmp):
    it = its[ismp]
    if ismp == 0:
      ax.semilogy(range(1, it + 1), res_norms[sum_it:(sum_it + it)], color=cols[ibJ], lw=.3, label='bJ%d' % nblock)
    else:
      ax.semilogy(range(1, it + 1), res_norms[sum_it:(sum_it + it)], color=cols[ibJ], lw=.3)
    sum_it += it
ax.grid()
ax.legend(ncol=2)
ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\|\mathbf{r}_k\|_2$')
pl.savefig('bJ-pcg.png', bbox_inches='tight')
