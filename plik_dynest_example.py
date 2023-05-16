import numpy as np
from samplers.dynamic_nested import DNS 
from likelihoods.planck_lik import Planck

pl = Planck()
lnL = pl.planck_lik 


paramranges = np.asarray([ [1.9e-9, 2.9e-9], [.94,.99], [0.03,.2], [0.020, 0.024], [0.11,0.13], [55,90], [.99, 1.01]])

# Initiate the sampler
dns = DNS(lnL,paramranges)

tol = .05
feedback_freq = 100
# Run the sampler
dns.dynamic_nested(tol,feedback_freq)

# plot(1000)

