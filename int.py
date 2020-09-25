import numpy as np
import matplotlib.pyplot as plt
import dynesty
from dynesty import plotting as dyplot
from likelihood import Likelihood 

data = np.load('data.npy')
# Define the dimensionality of our problem.
ndim = 2

# Define our 3-D correlated multivariate normal log-likelihood.
# C = np.identity(ndim)
# C[C==0] = 0.95
# Cinv = np.linalg.inv(C)
# lnorm = -0.5 * (np.log(2 * np.pi) * ndim +
#                 np.log(np.linalg.det(C)))

L = Likelihood(data)
# def loglike(x):
#     return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm

# Define our uniform prior via the prior transform.
def ptform(u):
    return 10. * u

# Sample from our distribution.
sampler = dynesty.NestedSampler(L.lnL, ptform, ndim,
                                bound='single', nlive=500)
sampler.run_nested(dlogz=0.1)
res = sampler.results


# Plot results.
# from dynesty import plotting as dyplot

# Plot a summary of the run.
# rfig, raxes = dyplot.runplot(res)

# Plot traces and 1-D marginalized posteriors.
# tfig, taxes = dyplot.traceplot(res)

# Plot the 2-D marginalized posteriors.
cfig, caxes = dyplot.cornerplot(res)

plt.show()
