import numpy as np
import likelihood as L


print( L.pseudoplanck([.320, .0221, .669, 3., .9632, .0524] ) )
# print( L.rosenbrock2d([1,1]) )
# print( L.rosenbrock2d([1,2]) )
# print( L.rosenbrock2d([2,1]) )
# print(L.lnL([5,3]))
# print(L.lnL([5,2]))
# print(L.lnL([15,13]))
# print(L.lnL([3,3]))

# integrate it over entire parameter space
# x = np.linspace(0,10,1000)
# y = np.linspace(0,10,1000)

# grid = np.zeros((1000,1000))
# for i,xi in enumerate(x):
#     print( i )
#     for j,yj in enumerate(y):
#         grid[i,j] =  L.lnL([xi,yj]) 

# grid = np.exp(grid)
# ix = np.zeros(1000)
# for i in range(1000):
#     ix[i] = np.trapz(grid[i,:],y)
#     print( i )
# I = np.trapz(ix,x)
# print( I )
# lnl = L.lnL(XX) #+ L.lnL(YY)
# print( lnl.shape )
