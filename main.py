import numpy as np
from samplers.metropolis import MetropolisHastings
from likelihood import Likelihood as lik 
from utils import plotter

data = np.load('data.npy')
# Initiate the sampler
MHsampler = MetropolisHastings(data)

# Define the sampler parameters
paramranges = np.asarray( [ [0.,25.],[0.,25.] ] )
num_samp = 5000

# Run the sampler
MHsampler.MH(num_samp,paramranges)

plotter.plot()

# burn out and sample selection
# burnout = 1000
# n_skip = 3
# post_samples = post_samples[burnout:]

# np.save('chains.npy', sampled_points)
# sampled_points = sampled_points[::n_skip]
# ################################
#  Plotting module. Should be generalized and shifted at some point

# print(post_samples)
# alpha = post_samples[:,0]
# beta = post_samples[:,1]
# print(alpha.mean(),beta.mean())
# import matplotlib.pyplot as plt
# # plt.plot(data, 'o', ms =0.5)

# plt.subplot(224)
# plt.hist(alpha, density=True, bins=50)
# x = np.linspace(alpha.min(),alpha.max(),100)
# plt.plot(x,norm.pdf(x,loc=alpha.mean(),scale=alpha.std()), color ='k')

# plt.subplot(221)
# # plt.hist(beta, density=True, orientation='horizontal', bins=50)
# plt.hist(beta, density=True, bins=50)
# y = np.linspace(beta.min(),beta.max(),100)
# plt.plot(y,norm.pdf(y,loc=beta.mean(),scale=beta.std()), color ='k')
# # plt.plot(norm.pdf(x,loc=beta.mean(),scale=beta.std()),x, color ='k')

# # plt.subplot(222)
# # xi = norm.pdf(x,loc=alpha.mean(),scale=alpha.std())
# # yi = norm.pdf(y,loc=beta.mean(),scale=beta.std())
# # z = np.outer(xi,yi)
# # print(z.shape)
# # plt.contour(x,y,z)

# plt.subplot(223)
# plt.scatter(alpha,beta,s=0.5)
# plt.axhline(beta.mean(), color='k', linewidth=0.5)
# plt.axvline(alpha.mean(), color='k', linewidth=0.5)
# # plt.show()
