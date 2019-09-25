import numpy as np
from numba import jit

def model(model_params):
	a=model_params[0]
	b=model_params[1]
	return (a + b**2)


def loglikelihood(data, model_params):
	chi_sq = np.sum((data-model(model_params))**2)
	sigma = np.std(data)
	C = sigma**2 * data.shape[0]
	
	# The loglikelihood
	lnL = - 0.5*chi_sq/C
	return lnL

# @jit()
def sampler(n_samp_max, data):
	new_lnlik = 0.0001 
	old_lnlik = 0.0
	accepted_points = []
	alpha = 5
	beta = 3
	while (len(accepted_points) < n_samp_max):
		# pick a random point to start in your model space
		delta = 0.01
		alpha = alpha + delta*np.random.standard_normal()
		beta = beta + delta*np.random.standard_normal()

		model_params = [alpha,beta]
		new_lnlik = loglikelihood(data,model_params)
		# print new_lnlik

		if (new_lnlik > old_lnlik) :# Accept this point 
			accepted_points.append(model_params)
			print '\naccepted'
			print len(accepted_points), model_params
			old_lnlik=new_lnlik
		else: # accept with probability L_new/L_old
			# print new_lnlik, old_lnlik,
			p = np.exp((new_lnlik-old_lnlik)/old_lnlik)
			# print '\nalpha = ', alpha, 'beta =', beta
			# print 'may accept with p = ', p
			if (np.random.random_sample() < p):
				accepted_points.append(model_params)
				# print 'accept'
				print len(accepted_points), model_params
				old_lnlik=new_lnlik
			else:
				# print 'rejected'
				pass

	return np.array(accepted_points)

npoints = 5000
# x = np.random.randn(npoints) # 1000 linearly spaced points in [0,10]
data_params = [5,3]
noise = 0.1*np.random.standard_normal(npoints)
data = np.repeat(model(data_params), npoints) + noise



# alpha = np.linspace(0,10,1000)
# lnL = np.empty(1000)
# for index,alp in enumerate(alpha):
# 	model_params=[alp,3]
# 	lnL[index] = loglikelihood(data, model_params)

# lnL -= lnL.max()


# # The sampling algorithm parameters
n_samp_max = 100
sampled_points = sampler(n_samp_max,data)

alpha = sampled_points[:,0]
beta = sampled_points[:,1]

import matplotlib.pyplot as plt
plt.hist(alpha, bins=20)
# plt.plot(alpha,np.exp(lnL))
plt.show()
