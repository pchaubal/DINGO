import numpy as np
from samplers.metropolis import MetropolisHastings
from likelihood import Likelihood as lik 
from utils.plotter import plot

data = np.load('data.npy')

# Define the sampler parameters
paramranges = np.asarray( [ [-10.,10.],[-10.,10.] ] )
num_samp = 50000

# Initiate the sampler
MHsampler = MetropolisHastings(data,paramranges)

initial_guess =np.asarray([5,3]) 
cov = .1*np.identity(2)
update_freq = 1000
# Run the sampler
MHsampler.MH(num_samp, initial_guess, cov, update_freq)

plot(burnout=10000)

