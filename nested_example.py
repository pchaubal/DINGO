import numpy as np
from samplers.nested_sampler import NestedSampler 
import likelihood
from utils.plotter import plot

# Define the likelihood
lnL = likelihood.gauss2d

# Define the sampler parameters
paramranges = np.asarray( [ [0.,10.],[0.,10.] ] )
nlive = 100

# Initiate the sampler
ns = NestedSampler(lnL,paramranges,nlive)

tol = .05
feedback_freq = 100
# Run the sampler
ns.Nested(tol,feedback_freq)

# plot(1000)

