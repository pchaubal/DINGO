import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
from chainconsumer import ChainConsumer
from likelihood import Likelihood
import matplotlib.pyplot as plt

class NestedSampler():

    def __init__(self,data,paramranges,n_live):
        self.data = data
        self.Lik = Likelihood(data)
        self.paramranges = paramranges
        self.ndims = self.paramranges.ndim
        self.n_live = n_live

    def Nested(self,tol,feedback_freq):
        # Sample the parameter space with n_live points
        live_pts = np.random.uniform(self.paramranges[:,0],self.paramranges[:,1],(self.n_live,self.ndims))
        
        # Evaluate the likelihood at live points
        live_lik = np.asarray([self.Lik.lnL(pt) for pt in live_pts])
        live_list = np.column_stack((live_pts,live_lik))
        
        # just some initialization stuff
        rejected_pts = np.zeros(3)
        X_now = 1.
        X = np.array([])
        deltaZ_percent = 100. # just an initialization value.
        deltaZmax = 100.
        lnZ = -np.inf
        n_evals = 0
        n_accepted = 0
        totals_evals = 0
        i=0
        while (deltaZmax>tol):
            # Sort the point with lowest likelihood value
            live_list = live_list[live_list[:,2].argsort()] 
            lowest_L_pt = live_list[0,:2]
            lowest_L = live_list[0,2]
            L_new = lowest_L

            # Select a new point with criteria: L(newpoint)>L(lowest L live point)
            found_newpt = False
            while not found_newpt:
                new_pt = self.propose(live_list)
                # Evaluate the likelihood of new point
                L_new = self.Lik.lnL(new_pt)
                n_evals += 1
                if ( L_new > lowest_L):
                    n_accepted += 1
                    found_newpt = True

            # Append old lowest point to rejected point list and replace it with new 
            if (i==0):
                rejected_pts = live_list[0]
            else:
                rejected_pts = np.vstack((rejected_pts,live_list[0]))
            
            # Remove the lowest likelihood point in favour of new point
            live_list[0] = np.asarray([new_pt[0],new_pt[1],L_new])
            live_list = live_list[live_list[:,2].argsort()] 
            
            # update the prior volume fraction by a deterministic approximate
            X = np.append(X,X_now)
            t1 = self.propose_t()
            X_now *= t1
#             X_now = X_now*(self.n_live-1)/self.n_live
            
            # Calculate evidence at every update_freq points
            # comapare it with previous evidence estimate
            if ((i+1)%feedback_freq==0):
                new_lnZ = np.log( -np.trapz(np.exp(rejected_pts[:,2]),X) ) 
                deltaZ_percent = 100*( np.exp(new_lnZ) - np.exp(lnZ) )/np.exp(lnZ)
                deltaZmax = 100*np.exp(live_list[-1,2])*X_now/np.exp(lnZ)
                print('\nLikelihood:', live_list[0,2])
                print( "Evidence:", np.exp(new_lnZ) )
                print( "Maximum possible Error %:",deltaZmax )
                print( "change in Evidence:", deltaZ_percent )
                print( "Acceptance ratio %:",100*n_accepted/n_evals)

                totals_evals += n_evals
                lnZ = new_lnZ
                n_accepted, n_evals = 0, 0
            i += 1
       ##---------------loop ends --------------##

        # Z = integral wrt X of lik 
        # negative sign is to correct for integrating along a decreasing axis
        Z =-np.trapz(np.exp(rejected_pts[:,2]),X) 
        # add contribution from remaining live points
        live_contribution =  np.sum(np.exp(live_list[:,2])*X_now/self.n_live)
        Z += live_contribution
#         Multiply the integral by volume of parameter space
        Z *= np.prod(np.diff(self.paramranges,axis=1))
        print( "\n\nZ = ",Z )
        print( "Actual error %:",100*np.abs(Z-.010724)/0.010724)
        print( "Total likelihood evaluations:", totals_evals )
        print( "Total iterations:",i )
#         posterior samples = Li wi/Z 
        weights = 0.5* (X[:-2] + X[2:])
        postlik = rejected_pts[1:-1,2] + np.log(weights) - np.log(Z)
        postsamples = np.column_stack((rejected_pts[1:-1,:],postlik))
        # add the livelist in post samples
        live_weight = X_now/self.n_live
        livesamples = np.column_stack((live_list[:,:],live_list[:,2]+ np.log(live_weight) - np.log(Z)))
        postsamples = np.vstack((postsamples,livesamples))
        # select equally weighted posterior samples
        K = np.max( postsamples[:,3])
        u = np.random.rand(len(postsamples))
        equal_wt_ind = np.where(np.log(u) < (postsamples[:,3]-K) )
        post_equal_weight = postsamples[equal_wt_ind]
        post_equal_weight = post_equal_weight[:,:2]
        
        print( np.std(post_equal_weight,axis=0) )
        c = ChainConsumer() 
#         c.add_chain(postsamples[:,:2],weights=postsamples[:,3])
        c.add_chain(post_equal_weight)
#         c.add_chain(post_equal_weight)
#         c.configure(kde=[True,False])
        c.plotter.plot(filename="example.jpg")
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(postsamples[:,0],postsamples[:,1],postsamples[:,2])
#         plt.plot(postsamples[:,3])
#         plt.plot(post_equal_weight[:,0],post_equal_weight[:,1],'o',markersize=1.0)
#         plt.plot(X,np.exp(rejected_pts[:,2]))
        plt.show()
   
    def propose(self,live_list):
        """ propose a new point """
        # Find the covmat with current set of live points
#         cov = np.cov(live_list[:,:2], rowvar=False)
#         Cinv = np.linalg.inv(cov)
        # rotate coordinates to principle axis

        # create an ellipsoid which touches max coordinate values
        # expand the limits by an enlargement factor
        # uniformly sample this extended ellipse
        mu = np.mean(live_list[:,:2],axis=0)
        sigma = np.std(live_list[:,:2],axis=0)
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
