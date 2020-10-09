import numpy as np
from samplers.metropolis import MetropolisHastings
from likelihood import Likelihood as lik 
from utils.plotter import plot

data = np.load('data.npy')

# Define the sampler parameters
paramranges = np.asarray( [ [0.,10] ,[0.,10] ] )
num_samp = 5000

# Initiate the sampler
MHsampler = MetropolisHastings(data,paramranges)

initial_guess = np.ones(2) 
cov =0.05*np.identity(2)
update_freq = 100
# Run the sampler
MHsampler.MH(num_samp, initial_guess, cov, update_freq)

plot(burnout=500)

