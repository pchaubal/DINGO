import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def init_animation():
    samples.set_data([],[])
    return samples 


def animate(i):
    global ax, fig
    n_walkers = 50
    samples.set_data(samp[:i*n_walkers,0], samp[:i*n_walkers,1])
    newpoint.set_data(samp[i*n_walkers:(i+1)*n_walkers,0], samp[i*n_walkers:(i+1)*n_walkers,1])
    time.sleep(.1)
    return samples 

samp = np.loadtxt('samples.txt')
fig = plt.figure()
ax = fig.add_subplot(xlim=(0,10), ylim=(0,10))
newpoint, = ax.plot([],[],'ro',ms=2)
samples, = ax.plot([],[],'bo',ms=0.5,linewidth=0.2,alpha=0.2)

ani = animation.FuncAnimation(fig,animate,frames=300,interval=1, init_func=init_animation)
plt.show()
