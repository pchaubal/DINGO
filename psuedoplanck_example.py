import numpy as np
from samplers.dynamic_nested import DNS 
import likelihood

# Define the likelihood
# lnL = likelihood.gauss2d
lnL = likelihood.pseudoplanck

# Define the sampler parameters
paramranges = np.asarray( [ [0.1,.5],[0.01,.04], [0.,1.], [1.,5.],[0.9,0.99],[0.03,.07] ] )

# Initiate the sampler
dns = DNS(lnL,paramranges)

tol = .05
feedback_freq = 100
# Run the sampler
dns.dynamic_nested(tol,feedback_freq)

# plot(1000)

