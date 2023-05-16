import numpy as np
from likelihood import Likelihood 
import matplotlib.pyplot as plt
from numba import jit


class AffineInv():
    def __init__(self,lnL,paramranges):
        self.posterior = None
        self.lnL = lnL 
        self.paramranges = paramranges
        self.ndims = len(self.paramranges)

    def afinv(self,n_walkers,n_steps):
        # Give the starting guesses uniformly form the param range, calculate the lnL for these
#         walkers = np.random.uniform(self.paramranges[:,0],self.paramranges[:,1],(n_walkers,self.ndims))
        #give the initial guess in a 1-sigma ball around the bestfit
        mean = np.load('../ML/planck_mean.npy')
        cov = np.load('../ML/planck_covmat.npy')
        p0 = np.random.multivariate_normal(mean, cov, size=n_walkers)
        p = np.zeros((n_walkers,self.ndims))
        p[:,:-1] = p0
        p[:,-1] = 1 +  0.005*np.random.randn(n_walkers)
        walkers = p

        lnLik = np.asarray([self.lnL(pt) for pt in walkers])
        
        self.samples = np.zeros((n_steps, n_walkers, self.ndims))
        for i_step in range(n_steps):
            print( i_step )
            for i_walker in range(n_walkers):
                #define the complementary ensemble by removing this walker 
                c_set = np.delete(walkers,i_walker, axis=0)
                walkers[i_walker], lnLik[i_walker] = self.update_walker(walkers[i_walker], lnLik[i_walker], c_set)
           
            self.samples[i_step] = walkers 
        ##----- nested loop ends-------- ##
        samples = self.samples.reshape( int(n_walkers*n_steps), self.ndims )
        np.savetxt("samples.txt",samples)
        return samples
    
    def update_walker(self, pt, lnLik, complementary_set):
        a = 2.0
        max_iter = 1000
        rng = np.random.default_rng()
        hook_pt = rng.choice(complementary_set, axis=0)

        found_newpt = False
        while (found_newpt==False): # run the loop until a new point is found
#         for i in range(max_iter):
            u = np.random.random()
            z = (a/(1+a))*(-1 + 2*u + a*u**2)
            newpt = pt + z*(hook_pt - pt)
            if self.is_within_boundaries(newpt):
                found_newpt = True
#                 break
        if not found_newpt:
            print( 'Did not find new point' )
        ###---------------###
        lnLik_new = self.lnL(newpt)
        if ( z**(self.ndims-1)*np.exp(lnLik_new - lnLik) > np.random.random() ): # Jump to the new point
            return newpt, lnLik_new 
        else:
            return pt, lnLik
    
    def is_within_boundaries(self,pt):
        return( np.all(pt> self.paramranges[:,0]) and np.all(pt < self.paramranges[:,1]) )
