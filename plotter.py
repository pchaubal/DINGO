import numpy as np
from chainconsumer import ChainConsumer


chains = np.load('chains.npy')
mean = chains.mean(axis=0)

c = ChainConsumer()
c.add_chain(chains, parameters=["$\\alpha$", "$\\beta$"])
c.plotter.plot(filename="example.jpg", figsize="column", truth=mean)

