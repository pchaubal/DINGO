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
#         L = np.dot(d,np.dot(self.Cinv,d.T))/(2*np.pi*self.detC)
        L =np.sum(d**2)
        return -np.log(L) 


