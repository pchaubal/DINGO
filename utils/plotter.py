import numpy as np
from chainconsumer import ChainConsumer
import pygtc
import matplotlib.pyplot as plt

def plot(burnout=1000):
    chain = np.loadtxt('samples.txt')[burnout:]
    mean = chain.mean(axis=0)
    c = ChainConsumer()
    c.add_chain(chain, parameters=["$\\alpha$", "$\\beta$"])
    c.plotter.plot(filename="example.jpg", figsize="column", truth=mean)
    # pygtc.plotGTC(chains=[chain])
    # plt.plot(chain[:,0],chain[:,1],'-o',markersize=1.0)
    plt.show()
