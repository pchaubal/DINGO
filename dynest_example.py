import numpy as np
from samplers.dynamic_nested import DNS 
import likelihood
from utils.plotter import plot

# Define the likelihood
lnL = likelihood.gauss2d

# Define the sampler parameters
paramranges = np.asarray( [ [0.,10.],[0.,10.] ] )

# Initiate the sampler
dns = DNS(lnL,paramranges)

tol = .05
feedback_freq = 100
# Run the sampler
dns.dynamic_nested(tol,feedback_freq)

# plot(1000)

