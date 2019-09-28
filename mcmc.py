import numpy as np
from numba import jit
from samplers import sampler_class

def model(model_params):
	a=model_params[0] + 0.01*np.random.standard_normal()
	b=model_params[1] + 0.01*np.random.standard_normal()
	return (2*a + b**2)


def loglikelihood(model_params):
	chi_sq = np.sum((data-model(model_params))**2)
	sigma = np.std(data)
	
	# The loglikelihood
	lnL = - 0.5*chi_sq/(len(data)*sigma**2)
	# print (lnL)
	return lnL

npoints = 5000
# x = np.random.randn(npoints) # 1000 linearly spaced points in [0,10]
data_params = [5,3]
noise = 0.10*np.random.standard_normal(npoints)
data = [model(data_params) for i in range(npoints)] + noise

# print(data)


# # The sampling algorithm parameters
n_samp_max = 1000
samples = sampler_class.MetropolisHastings(n_samp_max,loglikelihood)

alpha = samples[:,0]
beta = samples[:,1]
import matplotlib.pyplot as plt
plt.scatter(alpha,beta, s=0.5)
plt.axhline(3.0,color='k', label='true value')
plt.axvline(5.0,color='k')

plt.axvline(alpha.mean(),color='k', linestyle='--')
plt.axhline(beta.mean(),color='k', linestyle='--')
# plt.plot(data)
# plt.plot(alpha/5.0)
# plt.plot(beta/3.0)
# plt.hist(alpha,bins=20)
# plt.hist(beta,bins=20)

# plt.plot(alpha,np.exp(lnL))
plt.show()
