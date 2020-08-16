import numpy as np
from likelihood import Likelihood 

data = np.load('data.npy')
L = Likelihood(data)

print(L.lnL([5,3]))
print(L.lnL([5,2]))
print(L.lnL([15,13]))
print(L.lnL([3,3]))
