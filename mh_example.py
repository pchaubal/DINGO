import numpy as np
from samplers.metropolis import MetropolisHastings
from utils.plotter import plot
from likelihoods.planck_lik import Planck

# Define the likelihood
# data = np.load('data.npy')
# lnL = lik(data)
lnL = likelihood.gauss2d

# Define the sampler parameters
paramranges = np.asarray( [ [0.,10] ,[0.,10] ] )
num_samp = 10000

# Initiate the sampler
MHsampler = MetropolisHastings(lnL,paramranges)

initial_guess = np.ones(2) 
cov =0.1*np.identity(2)
update_freq = 100
# Run the sampler
MHsampler.MH(num_samp, initial_guess, cov, update_freq)

plot(burnout=1000)

