# --------------
#  GA_FUNCTIONS
# --------------
from NeuralNetwork import NeuralNetwork
import numpy as np
from engine import verlet, euler, Gravity2D, rigidBodyForce2D
import random
from numba import njit

@njit
def first_generation(pop_size:int, n_genes:int)->np.ndarray:
    # List of touples
    population = np.random.randn(pop_size, n_genes+1)
    population[:, 0] = -9999
    return population
 
@njit       
def new_generation(population, frac_to_replace)->np.ndarray:
    population = population[np.argsort(population[:,0]), :]
    to_replace = int(population.shape[0]*frac_to_replace)
    best = [population.shape[0]-to_replace, population.shape[0]-1 ]
    for i in range(to_replace):
        parent1 = population[random.randrange(best[0], best[1])]
        parent2 = population[random.randrange(best[0], best[1])]
        population[i] = child(parent1, parent2)
    return population

@njit
def child(parent1, parent2)->np.ndarray:
    child = np.zeros(parent1.shape[0])
    end = random.randrange(0, child.shape[0])
        
    child[:end] = parent1[:end]
    if np.random.random() < 0.05:
        # Mutation
        child[:end] += np.random.randn(end)

    child[end:] = parent2[end:]
    if np.random.random() < 0.05:
        child[end:] += np.random.randn(child.shape[0]-end)

    return child

def fitness(population, layers, save=False)-np.ndarray:
    for idx, ex in enumerate(population):
        genes = ex[1:]
        brain = NeuralNetwork(layers=layers, genes=genes)
        fitness = 0
        
        # --- Physics simulation ---
        dt = 0.001
        max_length = 61 # If you survive 60 seconds you won
        max_points = int(max_length/dt)
        bar_length = 1
        mc = 1000;           mp = 1

        # For now let'em start with everything at 0 Let's see what append      
        xc = np.zeros([max_points, 2]) # Center position [t, [x, y]]
        xp = np.zeros([max_points, 2]) # Pendulum position [t, [x, y]]
        vc = np.zeros([max_points, 2]);		vp = np.zeros([max_points, 2])
        Fp = np.zeros([max_points, 2]);		Fc = np.zeros([max_points, 2])
        
        # First time step is free fall
        xp[0, 0] = 0;   xp[0, 1] = bar_length
        vp[0, 0] = np.random.randn()*0.5
        
        Fp[0], _ = rigidBodyForce2D(xc=xc[0,:], xp=xp[0,:], length=bar_length)
        Fp[0, 1] += Gravity2D(m=mp)
        
        euler(x=xc, v=vc, F=Fc[0], m=mc, k=0, dt=dt)
        euler(x=xp, v=vp, F=Fp[0], m=mp, k=0, dt=dt)
        
        for i in range(1, max_points-1):
            state = np.array([
                        xc[i][0], # His position on x axes
                        (xc[i][0]-xc[i-1][0])/dt, # His velocity
                        xp[i][0]/xp[i][1], # angular position of the mass
                        (xp[i][0]/xp[i][1]-xp[i-1][0]/xp[i-1][1])/dt, # angular velocity of the mass
                    ])
            F_barCP, _ = rigidBodyForce2D(xc=xc[i], xp=xp[i], length=bar_length)
            Fp[i] = F_barCP
            Fp[i, 1] += Gravity2D(m=mp)
            Fc[i] = -Fp[i]
            Fc[i, 0] += mc*20*brain.forward_propagation(state)
        
            verlet(x=xc, F=Fc[i], m=mc, k=i, dt=dt)
            verlet(x=xp, F=Fp[i], m=mp, k=i, dt=dt)

            if xc[i+1, 0] < 3 and xc[i+1, 0] > -3 and xp[i+1, 1] > 0:
                fitness+=1
            else:
                break
        # --- end physics simulation ---
        if save:
            np.save("{:0>5d}-xp".format(fitness), xp)
            np.save("{:0>5d}-xc".format(fitness), xc)
        population[idx][0] = fitness