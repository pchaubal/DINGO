import numpy as np
from samplers.affineinv import AffineInv 
from utils.plotter import plot
from likelihoods.planck_lik import Planck

pl = Planck()
lnL = pl.planck_lik 

paramranges = np.asarray([ [1.9e-9, 2.7e-9], [.94,.97], [0.03,.1], [0.021, 0.023], [0.11,0.13], [60,80], [.996, 1.0]])
n_walkers = 100
n_steps = 500

af_sampler = AffineInv( lnL, paramranges ) 
samples = af_sampler.afinv(n_walkers,n_steps)
print( samples.shape )
np.save('planck_ml.npy', samples)
plot(samples, burnout=5000, skip=2)
