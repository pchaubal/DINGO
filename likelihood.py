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
    return -np.log(f + 0.01)

cov = np.loadtxt('planck_cov.txt')
def pseudoplanck(params):
#     cov = np.loadtxt('planck_cov.txt')
#     cov = np.identity(6)
    planck_means =np.asarray([.320, .0221, .669, 3., .9632, .0524])
    f = mvn(mean=planck_means, cov=cov)
    lnL = np.log(f.pdf(params))
    return lnL

def gauss2d(x):
    cov = 0.1*np.identity(2) 
    cov[cov==0] = 0.05
    mean = [5,5]
    f = mvn(mean=mean, cov=cov)
    lnL = np.log(f.pdf(x))
    return lnL

def gaussmix(params):
    cov1 = 5*np.identity(2) 
    cov2 = 0.1*np.identity(2) 
    mean = [5,5]
    f1 = mvn(mean=mean, cov=cov1)
    f2 = mvn(mean=mean, cov=cov2)
    lnL = np.log(f1.pdf(params) + 10*f2.pdf(params))
    return lnL
