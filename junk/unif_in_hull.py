from scipy.spatial import ConvexHull, Delaunay
import numpy as np
from numpy.linalg import det
from scipy.stats import dirichlet
import matplotlib.pyplot as plt



def dist_in_hull(points, n):
    dims = points.shape[-1]
    hull = points[ConvexHull(points).vertices]
    deln = points[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)    
    sample = np.random.choice(len(vols), size = n, p = vols / vols.sum())

    return np.einsum('ijk, ij -> ik', deln[sample], dirichlet.rvs([1]*(dims + 1), size = n))

points = np.random.rand(100,2)
samples = dist_in_hull(points,20000)

hull = ConvexHull(points)

plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'k--', lw=2)
plt.plot(points[:,0],points[:,1],'ok')
plt.plot(samples[:,0],samples[:,1],'or')
plt.show()
