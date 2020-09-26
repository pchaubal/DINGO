import numpy as np
from samplers.affineinv import AffineInv 
from utils.plotter import plot

data = np.load('data.npy')
paramranges = np.asarray( [ [0.,10.],[0.,10.] ] )
n_walkers = 50
n_steps = 500

af_sampler = AffineInv(data,paramranges) 
af_sampler.afinv(n_walkers,n_steps)

plot(500)
