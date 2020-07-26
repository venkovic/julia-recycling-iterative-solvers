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

nsmp = 10
nblocks = (10, 20, 30)
cols = ('tab:green', 'tab:blue', 'tab:red')

def get_rgb(s, nu, ind):
  rgb = 3 * [0]
  rgb[ind % 3] = s / (nu - 1.)
  return tuple(rgb)

fig, ax = pl.subplots(1, len(nblocks), figsize=(12, 3.5), sharey=True)
for ibJ, nblock in enumerate(nblocks):
  its = np.load('data/mrhs-bJ%d-incr-eigpcg.its.npy' % nblock)
  res_norms = np.load('data/mrhs-bJ%d-incr-eigpcg.res_norms.npy' % nblock)
  sum_it = 0
  for ismp in range(nsmp):
    col = get_rgb(ismp, nsmp, ibJ)
    it = its[ismp]
    ax[ibJ].semilogy(range(1, it + 1), res_norms[sum_it:(sum_it + it)], color=col, lw=1.3)
    sum_it += it
  ax[ibJ].grid()
  ax[ibJ].set_xlabel(r'$\mathrm{Solver\ iteration}\; j$')
  ax[ibJ].set_xticks([0, 25, 50, 75, 100])
  ax[ibJ].set_yticks([1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4])
  if ibJ == 0:
    ax[ibJ].set_ylabel(r'$\|\mathbf{r}_j\|_2$')
  ax[ibJ].set_xlim(-5, 105)
  ax[ibJ].set_title('bJ%d, $nvec$ = %d' % (nblock, nblock))
  pl.suptitle('Incremental eigPCG with a constant SPD matrix (1,000,000 DoFs) and multiple right-hand sides', y=1.08, fontsize=18)
pl.savefig('Example1_mrhs.png', bbox_inches='tight')
