# ------------------------------
#   DoublePendulumAnimation.py
# ------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set(rc={'axes.facecolor':'lightblue', 'figure.facecolor':'lightblue','figure.edgecolor':'black','axes.grid':False})

# -------------------------
#   ANIMATION PARAMETERS
# -------------------------
dt = 0.001
FPS = 30

# In order to save the .mp4 file some coded and
# library are needed with Anaconda 
# 	conda install -c conda-forge ffmpeg
save_video = not True


# --------------------------------------
#   FUNZIONI PER L'ANIMAZIONE
# --------------------------------------
def animate(i):
    i = int(i)
    print(i)

    pathX.append(xe[i,0])
    pathY.append(xe[i,1])
    
    x = [xc[i,0], xp[i,0], xe[i,0]]
    y = [xc[i,1], xp[i,1], xe[i,1]]
    points.set_data(x, y) 

    circleUP.set_data(x_circle+xc[i,0], y_circle+xc[i,1])
    circleDW.set_data(x_circle+xc[i,0], -y_circle+xc[i,1])
    
    path.set_data(pathX, pathY)
    
    return points, circleUP, circleDW, path


def init():
    ax.grid(True)
    
    ax.set_xlim([-3, 3])
    ax.set_xlabel('X')
    ax.set_ylim([-3, 3])
    ax.set_ylabel('Y')
    points.set_data([], [])
    circleUP.set_data([], [])
    circleDW.set_data([], [])
    
    del pathX[:]
    del pathY[:]
    path.set_data([], [])
    plt.tight_layout()
    return points, circleUP, circleDW, path

# --------------------------------------
#   MAIN
# --------------------------------------
xc = np.load("xc.npy")
xp = np.load("xp.npy")
xe = np.load("xe.npy")

fig, ax = plt.subplots(figsize=(5,5))
points, = ax.plot([], [], marker="o", lw=2)
circleUP, = ax.plot([], [], color="y")
circleDW, = ax.plot([], [], color="y")

pathX, pathY = [], []
path, = ax.plot(pathX, pathY, "b--", lw=0.5)

x_circle = np.linspace(-1, 1, 1000)
y_circle = -np.sqrt(1-x_circle**2)

skip = np.round( 1/(dt*FPS) )
frameIndices = np.arange(0, xc.shape[0], skip)


ani = animation.FuncAnimation(fig, animate, frames=frameIndices,
                              blit=False, interval=1/FPS*1000,
                              repeat=False, init_func=init)

if save_video:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('anim.mp4', writer=writer)

plt.show()
