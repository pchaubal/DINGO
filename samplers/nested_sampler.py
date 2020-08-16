import numpy as np
from likelihood import Likelihood

class Nested_Sampler(data):
    def __init__(self,data):
        self.data = data
        self.Lik = Likelihood(data)

    def Nested(self,n_live,tolerance):
        # Sample the parameter space with n_live points
        live_pt = np.random.uniform(lower,upper,n_live)
        
        # Evaluate the likelihood at live points

        # Remove the point with lowest likelihood value
        # Select a new point with criteria: L(newpoint)>L(lowest L live point)
        # if the lowest likelihood point hasnt changed by a value as high as the tolerance, finish the run 

