# ------------------------------
#     engine.py
# ------------------------------
import numpy as np
from numpy.linalg import norm
from numba import njit    

@njit
def rigidBodyForce2D(xc:np.ndarray, xp:np.ndarray, length:float) -> np.ndarray:
    '''
    Force generate by a stick of length "length"
    '''
    force2D = np.empty(2)
    distance = norm(xc-xp)
    
    dx = xp[0]-xc[0]
    dy = xp[1]-xc[1]

    force = -(distance-length)*np.exp(14+distance)
    force2D[0] = dx/distance * force
    force2D[1] = dy/distance * force

    return force2D, distance

@njit
def Gravity2D(m:float) -> float:
    gravityForce = -9.81*m
    return gravityForce

@njit
def verlet(x:np.ndarray, F:np.ndarray, m:float, k:int, dt:float) -> np.ndarray:
    '''
    Time evolution using Verlet Alghoritm. This is a pretty reliable method
    but it requires the first two steps
    '''
    # Verlet    
    x[k+1][0] = 2*x[k][0] - x[k-1][0] + (F[0]/m)*dt*dt
    x[k+1][1] = 2*x[k][1] - x[k-1][1] + (F[1]/m)*dt*dt
    
    return x[:][k+1]

@njit
def euler(x:np.ndarray, v:np.ndarray, F:np.ndarray,
          m:float, k:int, dt:float) -> np.ndarray:
    '''
    Time evolution using Euler Alghoritm
    '''
    v[k+1][0] = v[k][0]+(F[0]/m)*dt
    v[k+1][1] = v[k][1]+(F[1]/m)*dt
    
    # Euler-Cromer
    x[k+1][0] = x[k][0] + v[k+1][0]*dt
    x[k+1][1] = x[k][1] + v[k+1][1]*dt

    return x[:][k+1], v[:][k+1]