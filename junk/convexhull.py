from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np


points = np.random.rand(100, 2)   # 30 random points in 2-D
hull = ConvexHull(points)

print( hull.volume )
# print( hull.simplices )
# plt.plot(points[:,0], points[:,1], 'o')
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')


# plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
# plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
# plt.show()
