import numpy as np
from samplers import sampler_class

def model(model_params):
	a=model_params[0]
	b=model_params[1]
	return (a + a*b**2 + a**3)


def loglikelihood(model_params):
	sigma = np.std(data)
	chi_sq = np.sum((data-model(model_params))**2/sigma**2)
	# C = sigma**2
	
	# The loglikelihood
	lnL = - 0.5*chi_sq/5000.0
	return lnL

npoints = 5000
# x = np.random.randn(npoints) # 1000 linearly spaced points in [0,10]
data_params = np.asarray([5,3])
noise = 0.1*np.random.standard_normal(npoints)
data = [model(data_params + 0.1*np.random.standard_normal(data_params.shape)) for i in range(npoints)] + noise

#sampler parameters
n_samp_max = 1000
initial_guess = [3,2]
priors = [1.0,1.0]

sampler = sampler_class()
sampled_points = sampler.MetropolisHastings(n_samp_max, initial_guess, priors,loglikelihood)



# ################################
#  Plotting module. Should be generalized and shifted at some point


alpha = sampled_points[:,0]
beta = sampled_points[:,1]

import matplotlib.pyplot as plt
plt.subplot(224)
plt.hist(alpha, density=True, bins=20)

plt.subplot(221)
plt.hist(beta, density=True, orientation='horizontal', bins=20)

plt.subplot(222)
plt.scatter(alpha,beta,s=0.5)
plt.axhline(beta.mean(), color='k', linewidth=0.5)
plt.axvline(alpha.mean(), color='k', linewidth=0.5)
plt.show()
