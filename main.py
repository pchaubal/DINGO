import numpy as np
from samplers.nested_sampler import NestedSampler 
from likelihood import Likelihood as lik 
from utils.plotter import plot

data = np.load('data.npy')

# Define the sampler parameters
paramranges = np.asarray( [ [0.,10.],[0.,10.] ] )
nlive = 500

# Initiate the sampler
ns = NestedSampler(data,paramranges,nlive)

# Run the sampler
ns.Nested()

# plot(1000)

