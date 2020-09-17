import numpy as np
import os
from likelihood import Likelihood 

class MetropolisHastings():
    """Implements MH algorithm"""

    def __init__(self,data):
        self.posterior = None
        self.data = data
        self.Lik = Likelihood(data)

        # if previous file exists, delete it
        if os.path.isfile('samples.txt'):
            os.remove('samples.txt')
            print('Removed samples.txt from previous run')
        

	# @profile
    def MH(self,n_samples,paramranges):
        update_freq = 1000
        ndims = paramranges.ndim
        initial_guess = np.mean(paramranges,axis=1)

        lnL = self.Lik.lnL(initial_guess) 	#Initializing
        old_point = initial_guess

        proposal = 10.6*np.identity(ndims) # N(2.38^2*Sigma) = N(5.66*Sigma)
        samples = [initial_guess]
        n_accepted = 0
        for i in range(n_samples):
#             print(lnL)
            #Determine the new point
            new_pt =  old_point + np.dot(proposal,np.random.randn(ndims))
            while (np.any(new_pt>paramranges[:,1]) or np.any(new_pt<paramranges[:,0])): 
                new_pt =  old_point + np.dot(proposal,np.random.randn(ndims))


            lnL_new = self.Lik.lnL(new_pt)

            if ( np.exp(lnL_new - lnL) > np.random.random() ):
                # Accept the new point
                samples.append(new_pt)
                #Redefine the point
                old_point, lnL = new_pt, lnL_new
                # Update the number of accepted points
                n_accepted += 1
            else:
                samples.append(old_point)

            #Update the proposal matrix and print a feedback    
            if (i%update_freq==0):
                if i==0: continue # Dont do anything on first step

                samples = np.asarray(samples)
#                 print('\nLikelihood Evaluations:',i)
#                 print('Acceptance ratio:',n_accepted/update_freq)
                n_accepted=0
                # update the covmat
#                 print('Re-estimating the covariance matrix')
#                 covmat = np.cov(samples.T)/update_freq 
#                 proposal += (2.38**2)*covmat
#                 print(proposal)
                # write the data to a file
                with open('samples.txt','a') as f:
                    np.savetxt(f,samples)
                # Set samples to empty list as we have already saved these
                samples = []
                # print a feedback
#                 print('lnL=',lnL)
#                 print('Parameter Values:',old_point)

        return

    def tune(self):
        return
    def random_rotation(self):
        return

