# ------------------------------
#     engineDoublePendulum.py
# ------------------------------
import numpy as np
from numba import njit
from numpy.linalg import norm
import matplotlib.pyplot as plt
    
@njit(nopython=True)
def rigidBodyForce2D(xc:np.ndarray, xp:np.ndarray, length:float) -> np.ndarray:
    '''
    Force generate by a stick of length 1
    '''
    force2D = np.empty(2)
    distance = norm(xc-xp)
    
    dx = xp[0]-xc[0]
    dy = xp[1]-xc[1]

    force = -(distance-length)*np.exp(14+distance)
    force2D[0] = dx/distance * force
    force2D[1] = dy/distance * force

    return force2D, distance


@njit(nopython=True)
def Gravity2D(m:float) -> float:
    gravityForce = -9.81*m
    return gravityForce


@njit(nopython=True)
def verlet(x:np.ndarray, F:np.ndarray, m:float, k:int, dt:float) -> np.ndarray:
    '''
    Time evolution using Verlet Alghoritm. This is a pretty reliable method
    but it requires the first two steps
    '''
    # Verlet    
    x[k+1][0] = 2*x[k][0] - x[k-1][0] + (F[0]/m)*dt*dt
    x[k+1][1] = 2*x[k][1] - x[k-1][1] + (F[1]/m)*dt*dt
    
    return x[:][k+1]


@njit(nopython=True)
def euler(x:np.ndarray, v:np.ndarray, F:np.ndarray,
          m:float, k:int, dt:float) -> np.ndarray:
    '''
    Time evolution using Verlet Alghoritm. This is a pretty reliable methosd,
    but it requires the first two steps
    '''
    v[k+1][0] = v[k][0]+(F[0]/m)*dt
    v[k+1][1] = v[k][1]+(F[1]/m)*dt
    
    # Euler-Cromer
    x[k+1][0] = x[k][0] + v[k+1][0]*dt
    x[k+1][1] = x[k][1] + v[k+1][1]*dt

    return x[:][k+1], v[:][k+1]
  
if __name__ == "__main__":
    dt = 0.001
    length = 10 # sec
    points = int(length/dt)
    
    mc = 10000
    mp = 100
    me = 1 # massa estremo
    
    xc = np.zeros([points, 2]) # Center position [t, [x, y]]
    xp = np.zeros([points, 2]) # Pendulum position [t, [x, y]]
    xe = np.zeros([points, 2]) # Pendulum  extremum position [t, [x, y]]
    vc = np.zeros([points, 2]);		vp = np.zeros([points, 2]);		ve = np.zeros([points, 2])
    Fp = np.zeros([points, 2]);		Fc = np.zeros([points, 2]);		Fe = np.zeros([points, 2])
    
    distCP=np.zeros([points-1, 1])
    distPE=np.zeros([points-1, 1])

    xc[0, 0] = 0;       xc[0, 1] = 0;
    xp[0, 0] = 1;       xp[0, 1] = 0;
    xe[0, 0] = 1;       xe[0, 1] = 0.5;
    
    vc[0, 0] = 1;       vc[0, 1] = 0;
    vp[0, 0] = 0;       vp[0, 1] = 0;
    ve[0, 0] = 0;       ve[0, 1] = 0;
    
    Fp[0], distCP[0] = rigidBodyForce2D(xc=xc[0], xp=xp[0], length=1)
    Fe[0], distPE[0] = rigidBodyForce2D(xc=xp[0], xp=xe[0], length=0.5)
    Fp[0] = Gravity2D(m=mp)
    Fe[0] = Gravity2D(m=me)

    
    xc[1], _ = euler(x=xc, v=vc, F=Fc[0], m=mc, k=0, dt=dt)
    xp[1], _ = euler(x=xp, v=vp, F=Fp[0], m=mp, k=0, dt=dt)
    xe[1], _ = euler(x=xe, v=ve, F=Fe[0], m=me, k=0, dt=dt)
    
    for i in range(1, points-1):
        F_barPE, distPE[i] = rigidBodyForce2D(xc=xp[i], xp=xe[i], length=0.5)
        F_barCP, distCP[i] = rigidBodyForce2D(xc=xc[i], xp=xp[i], length=1)
        
        Fe[i] = Gravity2D(m=me) + F_barPE 
        Fp[i] = Gravity2D(m=mp) + F_barCP - Fe[i]
        Fc[i] = - Fp[i]

        Fc[i, 0] = Fc[i, 0] - 1*1e4*xc[i, 0]
        Fc[i, 1] = Fc[i, 1] - 1*1e6*xc[i, 1]
        
        
        #euler(x=xc, v=vc, F=Fc[i], m=mc, k=i, dt=dt)
        #euler(x=xp, v=vp, F=Fp[i], m=mp, k=i, dt=dt)
        
        verlet(x=xc, F=Fc[i], m=mc, k=i, dt=dt)
        verlet(x=xp, F=Fp[i], m=mp, k=i, dt=dt)
        verlet(x=xe, F=Fe[i], m=me, k=i, dt=dt)
        
    plt.figure(figsize=(12, 5))
    plt.plot(range(points-1), distPE-0.5, lw=0.5)
    plt.title("Differenza lunghezza PE")
    plt.xlabel("iteration")
    plt.ylabel(r"$\Delta$ $L_{PE}$")
    plt.grid(True)
    plt.tight_layout()
    
    plt.figure(figsize=(12, 5))
    plt.plot(range(points-1), distCP-1, lw=0.5)
    plt.title("Differenza lunghezza CP")
    plt.xlabel("iteration")
    plt.ylabel(r"$\Delta$ $L_{CE}$")
    plt.grid(True)
    plt.tight_layout()
    
    np.save("xc.npy", xc)    
    np.save("xp.npy", xp)       
    np.save("xe.npy", xe)
	
	