import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

def init_animation():
    samples.set_data([],[])
    return samples 


def animate(i):
    global ax, fig
    newpoint.set_data(samp[i,0], samp[i,1])
#     line.set_data(samp[i:i+1,0], samp[i:i+1,1])
    samples.set_data(samp[:i,0], samp[:i,1])
    return samples 

samp = np.loadtxt('samples.txt')[:,]
fig = plt.figure()
ax = fig.add_subplot(xlim=(0,10), ylim=(0,10))
e1 = Ellipse((5,5), width=1, height=0.5, angle=45,color='green', alpha=0.5)
e2 = Ellipse((5,5), width=2, height=1.0, angle=45,color='green', alpha=0.3)
ax.add_artist(e1)
ax.add_artist(e2)
newpoint, = ax.plot([],[],'ro',ms=3)
# line, = ax.plot([],[],'ok')
# line, = ax.plot([],[],'-k',linewidth=0.2)
samples, = ax.plot([],[],'-bo',ms=0.5,linewidth=0.2,alpha=0.5)

ani = animation.FuncAnimation(fig,animate,frames=1000,interval=1, init_func=init_animation)
ani.save('mh.gif',writer="ffmpeg",fps=30)
plt.show()
