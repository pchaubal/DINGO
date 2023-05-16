import numpy as np
import matplotlib.pyplot as plt
from samplers.single_ellipsoid import unif_ellipsoid
from chainconsumer import ChainConsumer
from utils import plotter

class DNS():

    def __init__(self,lnL,paramranges):
        self.lnL = lnL
        self.paramranges = paramranges
        self.ndims = paramranges.shape[0]
        self.n_evals = 0
    
    def dynamic_nested(self,tol,feedback_freq):
        """
        The main dynamic nested sampling function
        
        Parameters
        ----------
        tol: float
        the tolerance
        """
        # Run a low resolution static nested run for the baseline estimates
        post_ew, static_smp, post_wt, X_static = self.static_nested(tol,n_live=100)
#         plotter.plot(post_ew)
#         post_weighted_lik = np.exp(self.postsamples[:,-1])
        post_lik = static_smp[:,-1]
        post_wt = np.exp(post_wt)

        # define an importance function over the interval
        # Iz = (1. - Z/Zf)/np.trapz(1.-Z,X)
        fp = 1
        I = (fp)*post_wt/np.max(post_wt) #+ (1-fp)*Iz

        # Determine live points according to importance function
        # A +B*I -> atleast A points and max B at the highest resolution 
        num_live = 50 + 2000*I
        # make a live list
        n_live = 20
        live_list = self.make_live_list(n_live)
        rejected_pts = np.zeros(self.ndims+1)

        #---- Populating live list ----#
        Xnow =1.
        X = np.asarray([Xnow])
        # iterate over intervals of X
        for i,Xi in enumerate(X_static):
            Lmin = live_list[0,-1]
            if num_live[i]> n_live:
                while (n_live < num_live[i]):
                    # propose a point within the constrained volume
                    new_pt = self.propose_above_L(live_list,Lmin)
                    # Append it to live list
                    live_list = np.vstack((live_list,new_pt))
                    # Update the live points counter
                    n_live += 1
        #---- End of populating live list ----#
            live_list = live_list[live_list[:,-1].argsort()] 
        #---- Reduce the parameter volume ----#
            while (Lmin < post_lik[i+1]):
                rejected_pts = np.vstack((rejected_pts,live_list[0]))
                if (n_live > num_live[i]):
                    # Reduce without substitution
                    # remove Lmin from live_list
                    live_list = live_list[1:]
                    n_live -= 1
                else:
                    # Reduce exponentially by replacing Lmin point
                    new_pt = self.propose_above_L(live_list,live_list[0,-1])
                    # Substitute the lowest likelihood point by new point
                    live_list[0] = new_pt
                    live_list = live_list[live_list[:,-1].argsort()] 
                    Lmin = live_list[0,-1]
                
                    # Redefine parameter volume
                    # Reduction is same in both cases
                Xnow *= (n_live-1)/n_live 
                X = np.append(X,Xnow)
        print( rejected_pts.shape )
        print( "n_evals:", self.n_evals )

        post_ew, samples, post_wt = self.make_post_from_samples(rejected_pts,live_list,X) 
        
        print( post_ew.shape )
        plotter.plot(post_ew)

        return

    def static_nested(self,tol,n_live):

        live_list = self.make_live_list(n_live)
        
        # just some initialization stuff
        rejected_pts = np.zeros(self.ndims+1)
        X_now = 1.
        X = np.array([])
        maxZerr = 100.
        lnZ = -np.inf
        i=0
        while (maxZerr>tol):
            lowest_L_pt = live_list[0,:-1]
            lowest_L = live_list[0,-1]
            # Select a new point with criteria: L(newpoint)>L(lowest L live point)
            new_pt = self.propose_above_L(live_list,lowest_L)

            # Append old lowest point to rejected point list and replace it with new 
            if (i==0):
                rejected_pts = live_list[0]
            else:
                rejected_pts = np.vstack((rejected_pts,live_list[0]))
            
            # Substitute the lowest likelihood point by new point
            live_list[0] = new_pt 
            # sort live list
            live_list = live_list[live_list[:,-1].argsort()] 
            
            # update the prior volume fraction by a deterministic approximate
            X = np.append(X,X_now)
#             X_now *= self.propose_t(n-live)
            X_now *= (n_live-1)/n_live
            print( X_now, lowest_L )
            # Calculate the deltaZmax to decide whether to stop or not
            if (i!=0):
                Z =  -np.trapz(np.exp(rejected_pts[:,-1]),X) 
                lnZ = np.log( -np.trapz(np.exp(rejected_pts[:,-1]),X) ) 
                maxZerr = np.exp(live_list[-1,-1])*X_now*(1./np.exp(lnZ))
            # Update the iteration number 
            i += 1
       ##---------------loop ends --------------##

        post_ew, samples, post_wt = self.make_post_from_samples(rejected_pts,live_list,X) 
        print( 'Static run done. Summary for static run:')
        print( 'Equal Weight Samples:',post_ew.shape[0])
        print( 'Total samples (live + rejected):', samples.shape[0])
        print( 'Sampling efficiency:',f'{(samples.shape[0]-n_live)/self.n_evals:0.2f}' )
        return post_ew, samples, post_wt,X
        

    def propose_above_L(self,live_pts,Lmin):

        found_newpt = False
        while not found_newpt:
            new_pt = self.propose(live_pts)
            # Evaluate the likelihood of new point
            L_new = self.lnL(new_pt)
            self.n_evals += 1
            if ( L_new > Lmin):
                found_newpt = True
        print( self.n_evals )
        new_pt = np.append(new_pt,L_new)
        return new_pt


    def propose(self,live_pts):
        """ propose a new point """
        # Find the covmat with current set of live points
        cov = np.cov(live_pts[:,:-1], rowvar=False)
#         Cinv = np.linalg.inv(cov)
        
        # rotate coordinates to principle axis

        # create an ellipsoid which touches max coordinate values
        # expand the limits by an enlargement factor
        # uniformly sample this extended ellipse
        mu = np.mean(live_pts[:,:-1],axis=0)
#         sigma = np.std(live_pts[:,:-1],axis=0)
        success = False
        while not success: 
#             new_pt = mu + 4*sigma*(np.random.rand(self.ndims) - 0.5)
            new_pt = mu + unif_ellipsoid(1.2*cov) 
            success = self.is_within_boundaries(new_pt)
        return new_pt

    def is_within_boundaries(self,pt):
        return( np.all(pt> self.paramranges[:,0]) and np.all(pt < self.paramranges[:,1]) )
    
    def propose_t(self,nlive):
        """
        propose t from the pdf N*t**(N-1) with inverse probabilty transform
        the cdf is t**N, cdf inverse is t**(1/N)
        """
        u = np.random.random()
        t = u**(1./n_live)
        return t
   

    def make_post_from_samples(self,rejected_pts,live_pts,X):

        #--- Evidence Calculation ---#
        Z =-np.trapz(np.exp(rejected_pts[:,-1]),X) 
        live_lik = live_pts[:,-1]
#         live_contribution =  np.sum(live_lik*X[-1]/len(live_pts))
        live_wt = live_lik + np.log(X[-1]/len(live_pts))
        Z += np.sum(np.exp(live_wt))
        #--- Evidence Calculation ---#
        #--- Posterior weight calculation ---#
        weights = 0.5*( X[:-2] - X[2:] )
        rejected_lik = rejected_pts[1:-1,-1] # last column is logL
        rejected_wt = rejected_lik + np.log(weights) - np.log(Z)
#         post_wt_lik = rejected_lik + np.log(weights) - np.log(Z)
        # ---
        live_wt -= np.log(Z)
        samples = np.vstack((rejected_pts[1:-1],live_pts))
        sample_wt = np.concatenate((rejected_wt,live_wt))
        post_ew = self.make_equal_weight_samples(samples,sample_wt)
        return post_ew,samples,sample_wt

    def make_equal_weight_samples(self,samples,post_wt_lik):
        """
        Select equally weighted posterior samples

        Parameters
        ----------
        samples - an array of samples from nested sampling run with format 
        [sample_point,log(posterior_weight)]

        Returns
        -------
        equally weighted posterior samples
        """
        K = np.max( post_wt_lik)
        u = np.random.rand(len(samples))
        equal_wt_ind = np.where(np.log(u) < (post_wt_lik-K) )
        post_equal_weight = samples[equal_wt_ind]
        return post_equal_weight
    
    def make_live_list(self,n_live):
        # Sample the parameter space with n_live points
        live_pts = np.random.uniform(self.paramranges[:,0],self.paramranges[:,1],(n_live,self.ndims))
        
        # Evaluate the likelihood at live points
        live_lik = np.asarray([self.lnL(pt) for pt in live_pts])
        live_list = np.column_stack((live_pts,live_lik))
        # Sort the point with lowest likelihood value
        live_list = live_list[live_list[:,-1].argsort()] 
        return live_list
