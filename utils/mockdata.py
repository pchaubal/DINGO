import numpy as np
import matplotlib.pyplot as plt

def model(model_params):
    """
    We define a simple model where the observables are sum and product
    of the model parameters.
    """
    obs1 = model_params[:,0] + model_params[:,1]
    obs2 = model_params[:,0] - model_params[:,1]
    observables = np.vstack((obs1,obs2)).T
    return observables


npoints = 1000
n_obs = 2 #number of observations
data_params = np.asarray([5,3])
noise = 0.1*np.random.standard_normal([npoints,n_obs]) 
noisy_params = np.repeat(data_params[np.newaxis,...],npoints,axis=0) + 0.1*np.random.randn(npoints,len(data_params))
data = model(noisy_params) + noise
print(data.shape)
np.save('data.npy',data)

print(noisy_params.shape)
print(noisy_params)
# plt.plot(noisy_params[:,0],noisy_params[:,1],'o',markersize=1.)
plt.plot(data[:,0],data[:,1],'o',markersize=1.)
plt.show()
