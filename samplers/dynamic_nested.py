import numpy as np
import matplotlib.pyplot as plt

class DNS():

    def __init__(self,lnL,paramranges):
        self.lnL = lnL
        self.paramranges = paramranges
        self.ndims = self.paramranges.ndim
    
    def dynamic_nested(self,tol,feedback_freq):
        # run a low resolution static nested run for the baseline estimates
        self.static_nested(tol,n_live=100)
        post_weights = np.exp(self.postsamples[:,-1])
        # define an importance function over the interval
        # Iz = (1. - Z/Zf)/np.trapz(1.-Z,X)
        fp = 1
        I = (fp)*post_weights/np.max(post_weights) #+ (1-fp)*Iz
        # Determine points according to importance function
        # 1000 points at the highest resolution location
        num_live = 200*I
        # make a live list
        n_live = 100
        live_list = self.make_live_list(n_live)
        plt.plot(post_weights)
        plt.show()
        exit()
        #---- Populating live list ----#
        # iterate over intervals of X
        for i,Xi in enumerate(self.X):
            if num_live[i]> n_live:
                while (n_live < num_live[i]):
                    # propose a point within the constrained volume
                    new_pt = propose_in_constrained_X(live_list,)
                    # Append it to live list
                    np.append(live_list,new_pt)
                    # Update the live points counter
                    n_live += 1
        #---- End of populating live list ----#
        #---- Reduce the parameter volume ----#
            while (Xnow > self.X[i+1]):
                if (livepts > num_live[i]):
                    # Reduce without substitution
                    # remove Lmin from live_list
                    live_list = live_list[:-1]
                    n_live -= 1
                else:
                    # Reduce exponentially by replacing Lmin point
                    new_pt = propose_in_constrained_X()
                    # Substitute the lowest likelihood point by new point
                    live_list[0] = np.asarray([new_pt[0],new_pt[1],L_new])
                
                # Redefine parameter volume
                # Reduction is same in both cases
                Xnow *= (n_live-1)/n_live 
        return

    def static_nested(self,tol,n_live):
        self.n_live = n_live

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
            L_new = lowest_L
            # Select a new point with criteria: L(newpoint)>L(lowest L live point)
#             new_pt = self.propose_in_constrained_X(live_list,lowest_L)

            found_newpt = False
            while not found_newpt:
                new_pt = self.propose(live_list)
                # Evaluate the likelihood of new point
                L_new = self.lnL(new_pt)
                if ( L_new > lowest_L):
                    found_newpt = True

#             print( new_pt)
            # Append old lowest point to rejected point list and replace it with new 
            if (i==0):
                rejected_pts = live_list[0]
            else:
                rejected_pts = np.vstack((rejected_pts,live_list[0]))
            
            # Substitute the lowest likelihood point by new point
            live_list[0] = np.asarray([new_pt[0],new_pt[1],L_new])
            # sort live list
            live_list = live_list[live_list[:,-1].argsort()] 
            
            # update the prior volume fraction by a deterministic approximate
            X = np.append(X,X_now)
#             X_now *= self.propose_t()
            X_now *= (self.n_live-1)/self.n_live
            # Calculate the deltaZmax to decide whether to stop or not
            if (i!=0):
                Z =  -np.trapz(np.exp(rejected_pts[:,2]),X) 
                lnZ = np.log( -np.trapz(np.exp(rejected_pts[:,2]),X) ) 
                maxZerr = np.exp(live_list[-1,2])*X_now*(1./np.exp(lnZ))
            # Update the iteration number 
            i += 1
       ##---------------loop ends --------------##

        # Z = integral wrt X of lik 
        # negative sign is to correct for integrating along a decreasing axis
        Z =-np.trapz(np.exp(rejected_pts[:,-1]),X) 
        # add contribution from remaining live points
        live_lik = np.exp(live_list[:,-1])
        live_contribution =  np.sum(live_lik*X_now/self.n_live)
        Z += live_contribution
        # posterior samples = Li wi/Z 
        weights = 0.5*( X[:-2] - X[2:] )
        # leaving out first and last
        rejected_lik = rejected_pts[1:-1,-1]
        postlik = rejected_pts[1:-1,-1] + np.log(weights) - np.log(Z)
        self.postsamples = np.column_stack((rejected_pts[1:-1,:],postlik))
        # add the livelist in post samples
        live_weight = X_now/self.n_live
        livesamples = np.column_stack((live_list[:,:],live_list[:,-1]+ np.log(live_weight) - np.log(Z)))
        self.postsamples = np.vstack((self.postsamples,livesamples))
        self.X = X
        return
        
    def propose_in_constrained_X(self,live_pts,Lmin):

        found_newpt = False
        while not found_newpt:
            new_pt = self.propose(live_pts)
            # Evaluate the likelihood of new point
            L_new = self.lnL(new_pt)
            if ( L_new > Lmin):
                found_newpt = True
        return new_pt


    def propose(self,live_pts):
        """ propose a new point """
        # Find the covmat with current set of live points
#         cov = np.cov(live_list[:,:2], rowvar=False)
#         Cinv = np.linalg.inv(cov)
        # rotate coordinates to principle axis

        # create an ellipsoid which touches max coordinate values
        # expand the limits by an enlargement factor
        # uniformly sample this extended ellipse
        mu = np.mean(live_pts[:,:2],axis=0)
        sigma = np.std(live_pts[:,:2],axis=0)
        success = False
#         within_ellipsoid = False
        while not success: 
            new_pt = mu + 4*sigma*(np.random.rand(self.ndims) - 0.5)
#             within_ellipsoid = ( (new_pt - mu)@Cinv@(new_pt - mu).T - 15.0 ) < 0
#             success = within_ellipsoid and self.is_within_boundaries(new_pt)
            success = self.is_within_boundaries(new_pt)
        return new_pt

    def is_within_boundaries(self,pt):
        return( np.all(pt> self.paramranges[:,0]) and np.all(pt < self.paramranges[:,1]) )
    
    def propose_t(self):
        """
        propose t from the pdf N*t**(N-1) with inverse probabilty transform
        the cdf is t**N, cdf inverse is t**(1/N)
        """
        u = np.random.random()
        t = u**(1/self.n_live)
        return t
    
    def make_equal_weight_samples(samples):
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
        K = np.max( samples[:,-1])
        u = np.random.rand(len(samples))
        equal_wt_ind = np.where(np.log(u) < (samples[:,-1]-K) )
        post_equal_weight = samples[equal_wt_ind]
        post_equal_weight = post_equal_weight[:,:-1]
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
