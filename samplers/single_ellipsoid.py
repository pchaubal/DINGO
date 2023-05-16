import numpy as np

def uniform_unit_sphere(ndim):
    """
    Taken from this math stack post:

    """
    x = np.random.randn(ndim)
    x_hat = x/np.linalg.norm(x)
    pt = x_hat*(np.random.rand())**(1/ndim)
    return pt

def unif_ellipsoid(cov):
    ndim = cov.shape[0]
    a = np.zeros((ndim,ndim))
    np.fill_diagonal(a,1e-15)
    cov += a
    cholesky = np.linalg.cholesky(cov)
    pt = np.dot(cholesky,uniform_unit_sphere(ndim))
    return pt


