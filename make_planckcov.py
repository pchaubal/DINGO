import numpy as np

c = np.loadtxt('./planck_TTTEEE_lowl_lowE.txt')[:,[31,2,29,6,7,5]]
c[:,2] *= 1e-2

cov = np.cov(c.T)
cov2 = np.diag(np.diag(cov))

np.savetxt('planck_cov.txt',cov)
