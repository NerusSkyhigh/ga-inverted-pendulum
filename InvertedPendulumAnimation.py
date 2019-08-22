# --------------------------------
#   InvertedPendulumAnimation.py
# --------------------------------
import numpy as np
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
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
    cart.set_xy(  [xc[i,0]-0.5, xc[i,1]-0.25])
    wheelL.center = [xc[i,0]-0.3, xc[i,1]-0.25]
    wheelR.center =[xc[i,0]+0.3, xc[i,1]-0.25]

    x = [xc[i,0], xp[i,0]]
    y = [xc[i,1], xp[i,1]]
    points.set_data(x, y)
    
    return points, cart, wheelL, wheelR

def init():
    ax.set_xlim([-3, 3]);           ax.set_ylim([-1, 2])
    ax.set_xlabel('X');             ax.set_ylabel('Y')
    x_ticks = np.arange(-3, 5, 1);  y_ticks = np.arange(-1, 3, 1)
    ax.set_xticks(x_ticks);         ax.set_yticks(y_ticks)
    ax.grid(True)    
    plt.tight_layout()

    points.set_data([], [])
    ax.add_patch(cart)
    ax.add_patch(wheelL)
    ax.add_patch(wheelR)

    return points, cart, wheelL, wheelR

# --------------------------------------
#   MAIN
# --------------------------------------
fileNum = "46215"
xc = np.load(fileNum+"-xc.npy")
xp = np.load(fileNum+"-xp.npy")

fig, ax = plt.subplots(figsize=(6,3))

cart = Rectangle((-0.5,-0.25), width=1, height=0.5, fc='y',
                          linewidth=1, edgecolor='r',facecolor='none')
wheelL = Circle((-0.30,-0.30), radius=0.15, fc='y',
                          linewidth=1,edgecolor='r',facecolor='none')
wheelR = Circle((0.30,-0.30), radius=0.15, fc='y',
                          linewidth=1,edgecolor='r',facecolor='none')

points, = ax.plot([], [], marker="o", lw=2)

skip = np.round( 1/(dt*FPS) )

frameIndices = np.arange(0, xc.shape[0], skip)

frameIndices = np.arange(0, 46215, skip)


ani = animation.FuncAnimation(fig, animate, frames=frameIndices,
                              blit=False, interval=1000/FPS,
                              repeat=False, init_func=init)

if save_video:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('anim.mp4', writer=writer)

plt.show()
