import numpy as np

class sampler_class(object):
	"""docstring for sampler_class"""
	def __init__(self, arg):
		pass

	def MetropolisHastings(n_samples, loglikelihood):
		initial_guess = [4,2]
		old_lnlik = loglikelihood(initial_guess) 	#Initializing
		accepted_points = []
		
		while (len(accepted_points) < n_samples):
			old_point = np.array(initial_guess)
			delta = min(1.0,-old_lnlik) # the size of step
			new_point = old_point + delta*np.random.standard_normal(old_point.shape)
			# alpha = alpha + delta*np.random.standard_normal()
			# beta = beta + delta*np.random.standard_normal()

			new_lnlik = loglikelihood(new_point)
			# print new_lnlik

			if (new_lnlik > old_lnlik) :# Accept this point 
				accepted_points.append(new_point)
				# print ('\naccepted')
				print('\nL=',new_lnlik)
				print (len(accepted_points), new_point)
				print(delta)

				old_lnlik=new_lnlik
				old_point = new_point
			else: # accept with probability L_new/L_old
				p = np.exp(new_lnlik-old_lnlik)
				# p=0.0

				# if (p!=0):
				# 	print ('newL, oldL:',new_lnlik, old_lnlik,)
				# 	print ('may accept with p = ', p)

				if (np.random.random_sample() < p):
					accepted_points.append(new_point)
					# print ('accept')
					print('\nL=',new_lnlik)
					print (len(accepted_points), new_point)
					print(delta)
					old_lnlik=new_lnlik
					old_point=new_point
				else:
					# print 'rejected'
					pass

		return np.array(accepted_points)
