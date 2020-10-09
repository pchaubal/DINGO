import numpy as np
from samplers.nested_sampler import NestedSampler 
from likelihood import Likelihood as lik 
from utils.plotter import plot

data = np.load('data.npy')

# Define the sampler parameters
paramranges = np.asarray( [ [0.,10.],[0.,10.] ] )
nlive = 5000

# Initiate the sampler
ns = NestedSampler(data,paramranges,nlive)

tol = .1
feedback_freq = 100
# Run the sampler
ns.Nested(tol,feedback_freq)

# plot(1000)

