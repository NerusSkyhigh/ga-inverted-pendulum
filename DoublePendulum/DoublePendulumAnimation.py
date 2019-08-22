# ------------------------------
#   DoublePendulumAnimation.py
# ------------------------------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

sns.set(rc={'axes.facecolor':'lightblue',
            'figure.facecolor':'lightblue',
            'figure.edgecolor':'black',
            'axes.grid':False})

# -------------------------
#   ANIMATION PARAMETERS
# -------------------------
dt = 0.001 # Must be the same as engineDoublePendulum
FPS = 30

# In order to save the .mp4 file some codex and
# library are needed. You can install with Anaconda
# 	conda install -c conda-forge ffmpeg
save_video = True


# ------------------------
#   ANIMATION FUNCTIONS
# ------------------------
def animate(i):
    i = int(i)
    #print(i)

    pathX.append(xe[i,0])
    pathY.append(xe[i,1])
    path.set_data(pathX, pathY)
    
    x = [xc[i,0], xp[i,0], xe[i,0]]
    y = [xc[i,1], xp[i,1], xe[i,1]]
    points.set_data(x, y) 
    circle.center = (xc[i,0], xc[i,1])
    
    return points, path, circle


def init():
    ax.grid(True)
    
    ax.set_xlim([-3, 3])
    ax.set_xlabel('X')
    ax.set_ylim([-3, 3])
    ax.set_ylabel('Y')
    ax.add_patch(circle)
    points.set_data([], [])
    
    del pathX[:]
    del pathY[:]
    path.set_data([], [])
    plt.tight_layout()
    return points, path, circle

# ------------
#     MAIN
# ------------
xc = np.load("xc.npy")
xp = np.load("xp.npy")
xe = np.load("xe.npy")

fig, ax = plt.subplots(figsize=(5,5))
points, = ax.plot([], [], marker="o", lw=2)
circle = Circle( (0,0), radius=1, linewidth=1,
                edgecolor='y',facecolor='none')

pathX, pathY = [], []
path, = ax.plot(pathX, pathY, "b--", lw=0.5)

skip = np.round( 1/(dt*FPS) )
frameIndices = np.arange(0, xc.shape[0], skip)


ani = animation.FuncAnimation(fig, animate, frames=frameIndices,
                              blit=False, interval=1000/FPS,
                              repeat=False, init_func=init)

if save_video:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('anim.mp4', writer=writer)

plt.show()
