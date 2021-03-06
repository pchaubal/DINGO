import numpy as np
from scipy.stats import multivariate_normal as mvn

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

    def rosenbrock2d(self,params):
        x = params[0]
        y = params[1]

        f = (1. - x**2.) + 100.*(y - x**2.)**2.
#         print( "rosenbrock lik:", -np.log(f) )
        return -np.log(f + 0.01)

    def pseudoplanck(self,params):
        cov = np.loadtxt('planck_covmat.txt')
#         cov = np.identity(7)*np.asarray([.01,.001,.01,1e-9,.001,.001,.01])
#         cov = np.identity(7)*0.01
#         print( cov )
#         import matplotlib.pyplot as plt
#         plt.imshow(cov); plt.show()
#         exit()
#         print(np.linalg.det(cov))
#         planck_means = [.320, .0221, .669, 2.09219606e-9, .9632, .0524, .8126]
        planck_means = 0.05*np.ones(7)
#         planck_means = [5,5,5,5,5,5,5]
        f = mvn(mean=planck_means, cov=cov)
        lnL = np.log(f.pdf(params))
        return lnL
    
    def gauss2d(self,params):
        cov = [[0.1,0.05],[0.05,0.1]]
        mean = [5,5]
        f = mvn(mean=mean, cov=cov)
        lnL = np.log(f.pdf(params))
        return lnL
