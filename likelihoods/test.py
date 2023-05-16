from planck_lik import Planck
import numpy as np
import matplotlib.pyplot as plt


pl = Planck()

A_planck = 1. + 0.01*np.random.randn()
As =2.1e-9 
ns = .960
tau = 0.051
ombh2 = 0.0221
omch2 = 0.112 
H0 = 70. 

params = [As, ns, tau, ombh2, omch2, H0, A_planck]
params = np.asarray(params)

for i in range(1000):
    lnL = pl.planck_lik(params)
    print( lnL )
