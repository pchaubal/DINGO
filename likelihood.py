import numpy as np

class Likelihood:
    def __init__(self,data):
        self.C = np.cov(data.T)
        self.detC = np.linalg.det(self.C)
        self.Cinv = np.linalg.inv(self.C)
        self.data = data
    
    @staticmethod 
    def model(model_params):
        """
        We define a simple model where the observables are sum and product
        of the model parameters.
        """
        obs1 = model_params[0] + model_params[1]
        obs2 = model_params[0] - model_params[1] 
        observables = np.asarray([obs1,obs2])
        return observables

    def lnL(self,model_params):
        d = self.data - self.model(model_params)
        chisq =  np.sum( d*(self.Cinv @ d.T).T )/1e6
        chisq /= (2*np.pi*self.detC)
        lnL = -0.5*chisq
        return lnL 


