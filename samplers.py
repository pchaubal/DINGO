import numpy as np
from numba import jit

class sampler_class(object):
	"""docstring for sampler_class"""
	def __init__(self):
		pass

	# @jit
	def MetropolisHastings(self,n_samples,initial_guess, loglikelihood):
		print_after_n = 5000

		step_size = np.repeat([0.05],len(initial_guess)) # Initialize the step size to be 0.1 for all parameter.

		old_lnlik = loglikelihood(initial_guess) 	#Initializing
		accepted_points = []
		
		old_point = initial_guess

		# print('Initializing MH sampling')
		while (len(accepted_points) < n_samples):

			# This is the proposal density
			# delta = min(1.0,-old_lnlik) # the size of step
			# new_point = old_point + delta*np.random.standard_normal(old_point.shape)
			new_point = old_point+ step_size*np.random.standard_normal(old_point.shape)
			new_lnlik = loglikelihood(new_point)
			
			ln_p = new_lnlik-old_lnlik

			if ( ln_p > np.log(np.random.random_sample()) ):
				# Accept the new point
				accepted_points.append(new_point)
				old_lnlik = new_lnlik
				old_point = new_point

				if (len(accepted_points)%print_after_n==0):
					print('\nL=',new_lnlik)
					print (len(accepted_points), new_point)
			else:
				accepted_points.append(old_point) # Accept the old point

		return np.array(accepted_points)


	def HamiltonianMC(self,n_samples,initial_guess, priors, loglikelihood):
		return accepted_points
