import numpy as np
from samplers.affineinv import AffineInv 
from utils.plotter import plot

data = np.load('data.npy')
paramranges = np.asarray( [ [0.,20.],[0.,20.] ] )

af_sampler = AffineInv(data) 
af_sampler.afinv(50,paramranges)

plot(2000)
