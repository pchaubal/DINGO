import numpy as np
from likelihood import Likelihood 
import matplotlib.pyplot as plt


class AffineInv():
    def __init__(self,data,paramranges):
        self.posterior = None
        self.data = data
        self.Lik = Likelihood(data)
        self.paramranges = paramranges

    def afinv(self,n_walkers,n_steps):
        a = 2.0
        walkers = list(range(n_walkers))
        ndims = self.paramranges.ndim

        points = np.random.uniform(self.paramranges[:,0],self.paramranges[:,1],(n_walkers,ndims))
        lnL = np.asarray([self.Lik.lnL(pt) for pt in points])
        samples = None
        for iteration in range(n_steps):
            for i in range(n_walkers):
                #draw walker from complementary ensemble
                c_set = walkers.copy()
                c_set.pop(i)
                k = np.random.choice(c_set)

                found_newpt = False
                while (found_newpt==False):
                    u = np.random.random()
                    z = (a/(1+a))*(-1 + 2*u + a*u**2)
                    new_point = points[i] + z*(points[k] - points[i])
                    if (np.all(new_point<self.paramranges[:,1]) and np.all(new_point>self.paramranges[:,0])):
                        found_newpt = True
                lnL_new = self.Lik.lnL(new_point)
                if (z**(ndims-1)*np.exp(lnL_new - lnL[i]) > np.random.random() ):
                    # Accept and update
                    points[i] = new_point
                    lnL[i] = lnL_new
#                     print( "accepted" )
            
            if samples is not None:
                samples = np.vstack((samples,points))
            else:
                samples = points

        np.savetxt("samples.txt",samples)
#         plt.plot(samples[:,0],samples[:,1],'o',markersize=1.0)
#         plt.show()
    
