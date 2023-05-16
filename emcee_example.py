from utils.plotter import plot
import matplotlib.pyplot as plt
import numpy as np
from likelihoods.planck_lik import Planck
import emcee
from chainconsumer import ChainConsumer

pl = Planck()
lnL = pl.planck_lik 

ndim = 21
nwalkers = 100

#give the initial guess in a 1-sigma ball around the bestfit
# mean = np.load('../ML/planck_mean_theta.npy')
# cov = np.load('../ML/planck_covmat_theta.npy')
c = np.loadtxt('../clustercosmo/planck_chains/plikHM_TT_lowl_lowE/planck_TT_lowl_lowE.txt')
c = c[:,[37,7,5,2,3,51,  9,10,11,12,13,14,15,16,17,18,19,20,21,22,8]]
mean = c.mean(axis=0)
cov = np.cov(c.T)
p = np.random.multivariate_normal(mean, cov, size=nwalkers)
p[:,0] *= 1e-9
# p = np.zeros((nwalkers,ndim))
# p[:,:6] = p0
# p[:,6:] = 1. + np.random.randn(nwalkers,ndim-6) 
# p = np.load('./planck_ml.npy')[:nwalkers:,:]

# test_pt = np.array([2.19e-9, 0.961, 0.052, 0.0221, 0.12, 1.047, 0.99998])
# test_pt = np.zeros(7) 
# test_pt[:6] = mean
# test_pt[6] = 0.998
# lntest = np.zeros(100)
# Atest = np.linspace(0.999,1.002,100)
# for i,A in enumerate(Atest):
#     test_pt[6] = A
#     print( lnL(test_pt) )
#     lntest[i] = lnL(test_pt)

# plt.plot(Atest, lntest )
# plt.show()
# exit()
#     print(i, lnL(test_pt))
# print( test_pt )
# print( lnL(test_pt) )
# exit()
# initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnL)

#burnout run
state = sampler.run_mcmc(p, 100)
sampler.reset()

print( 'burnout done' )

#actual run
sampler.run_mcmc(state, 500);

samples = sampler.get_chain(flat=True)

samples[:,0] *= 1e9
np.save('planck_ml.npy',samples)
c = ChainConsumer()
names = ['$10^{10}A_s$','$ n_s $', '$ \\tau $', '$\Omega_bh^2$', '$\Omega_ch^2$','$H_0$']
planck_chain = np.loadtxt('/home/pc/codes/clustercosmo/planck_chains/plikHM_TT_lowl/plik_TT_lowl.txt')[:,[37,7,5,2,3,51]]
c.add_chain(samples[:,:6], parameters=names, name='ML trained spectra')
c.add_chain(planck_chain, parameters=names, name='Planck TT ')
c.plotter.plot(filename="example.jpg", figsize="column" )
plt.show()
