import numpy as np
from scipy.stats import norm
from samplers import sampler_class

def model(model_params):
	a=model_params[0]
	b=model_params[1]

	# observables = np.asarray([a+b, a-b, a*b])
	observables = np.asarray([a+b, np.log(a*b)])


	
	return observables


def loglikelihood(model_params):
	C = np.cov(data.T)
	Cinv = np.linalg.inv(C)
	d = data - model(model_params)

	lnL = -0.5*np.einsum('ij,jl,li ->',d,Cinv,d.T)

	return lnL

npoints = 1000
n_obs = 2 #number of observations
# x = np.random.randn(npoints) # 1000 linearly spaced points in [0,10]
data_params = np.asarray([5,3])
noise = 20.0*np.random.standard_normal([npoints,n_obs])
data = np.asarray([model(data_params + 0.1*np.random.standard_normal(data_params.shape)) for i in range(npoints)]) + noise

#sampler parameters
n_samp_max = 100000
initial_guess = np.asarray([3.0,2.])
# priors = [0.05,0.05]

sampler = sampler_class()
sampled_points = sampler.MetropolisHastings(n_samp_max, initial_guess, loglikelihood)


# burn out and sample selection
burnout = 1000
n_skip = 3
sampled_points = sampled_points[burnout:]

np.save('chains.npy', sampled_points)
sampled_points = sampled_points[::n_skip]
# ################################
#  Plotting module. Should be generalized and shifted at some point


alpha = sampled_points[:,0]
beta = sampled_points[:,1]

import matplotlib.pyplot as plt
# plt.plot(data, 'o', ms =0.5)

plt.subplot(224)
plt.hist(alpha, density=True, bins=50)
x = np.linspace(alpha.min(),alpha.max(),100)
plt.plot(x,norm.pdf(x,loc=alpha.mean(),scale=alpha.std()), color ='k')

plt.subplot(221)
# plt.hist(beta, density=True, orientation='horizontal', bins=50)
plt.hist(beta, density=True, bins=50)
y = np.linspace(beta.min(),beta.max(),100)
plt.plot(y,norm.pdf(y,loc=beta.mean(),scale=beta.std()), color ='k')
# plt.plot(norm.pdf(x,loc=beta.mean(),scale=beta.std()),x, color ='k')

# plt.subplot(222)
# xi = norm.pdf(x,loc=alpha.mean(),scale=alpha.std())
# yi = norm.pdf(y,loc=beta.mean(),scale=beta.std())
# z = np.outer(xi,yi)
# print(z.shape)
# plt.contour(x,y,z)

plt.subplot(223)
plt.scatter(alpha,beta,s=0.5)
plt.axhline(beta.mean(), color='k', linewidth=0.5)
plt.axvline(alpha.mean(), color='k', linewidth=0.5)
# plt.show()
