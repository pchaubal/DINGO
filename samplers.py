import numpy as np

class sampler_class(object):
	"""docstring for sampler_class"""
	def __init__(self, arg):
		pass

	def MetropolisHastings(n_samples, loglikelihood):
		new_lnlik = 0.0001  #Initialize the value to some small value
		old_lnlik = 0.0 	#Initializing
		accepted_points = []
		alpha = 5	#First guess
		beta = 3	#First guess
		while (len(accepted_points) < n_samples):
			# pick a random point to start in your model space
			delta = 0.01 # the size of step
			alpha = alpha + delta*np.random.standard_normal()
			beta = beta + delta*np.random.standard_normal()

			model_params = [alpha,beta]
			new_lnlik = loglikelihood(model_params)
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
