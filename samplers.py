import numpy as np

class sampler_class(object):
	"""docstring for sampler_class"""
	def __init__(self):
		pass

	def MetropolisHastings(self,n_samples,initial_guess,priors, loglikelihood):
		print_after_n = 100

		old_lnlik = loglikelihood(initial_guess) 	#Initializing
		accepted_points = []
		
		old_point = np.array(initial_guess)
		while (len(accepted_points) < n_samples):

			# This is the proposal density
			# delta = min(1.0,-old_lnlik) # the size of step
			# new_point = old_point + delta*np.random.standard_normal(old_point.shape)
			new_point =old_point+ priors*np.random.standard_normal(old_point.shape)

			


			new_lnlik = loglikelihood(new_point)
			
			# print('\n',old_point,new_point)
			# print(old_lnlik, new_lnlik)


			p = np.exp(new_lnlik-old_lnlik)
			pi = min(1.0,p)

			if (np.random.random_sample() < pi):
				# Accept
				accepted_points.append(new_point)
				old_lnlik = new_lnlik
				old_point = new_point

				if (len(accepted_points)%print_after_n==0):
					print('\nL=',new_lnlik)
					print (len(accepted_points), new_point)
			else:
				# Accept the old point
				accepted_points.append(old_point)



			# if (new_lnlik > old_lnlik) :# Accept this point 
			# 	accepted_points.append(new_point)
			# 	# print ('\naccepted')
			# 	print('\nL=',new_lnlik)
			# 	print (len(accepted_points), new_point)
			# 	print(delta)

			# 	old_lnlik=new_lnlik
			# 	old_point = new_point
			# else: # accept with probability L_new/L_old
			# 	p = np.exp(new_lnlik-old_lnlik)
			# 	# p=0.0

			# 	# if (p!=0):
			# 	# 	print ('newL, oldL:',new_lnlik, old_lnlik,)
			# 	# 	print ('may accept with p = ', p)

			# 	if (np.random.random_sample() < p):
			# 		accepted_points.append(new_point)
			# 		# print ('accept')
			# 		print('\nL=',new_lnlik)
			# 		print (len(accepted_points), new_point)
			# 		print(delta)
			# 		old_lnlik=new_lnlik
			# 		old_point=new_point
			# 	else:
			# 		# print 'rejected'
			# 		pass

		return np.array(accepted_points)
