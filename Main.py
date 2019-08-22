# ------------
#   MAIN.py
# ------------
import numpy as np
from ga_functions import first_generation, fitness, new_generation


population_size = 200
iteration = 10_000
# The neural network is made of 6 input and 1 output
n_genes = (4*6)+(6*6)+(6*6)+(6*6)+(6*1) # 4
layers = [4, 6, 6, 6, 6, 1]

# If you want to train the alghoritm from zero uncomment this line
# population = first_generation(population_size, n_genes)

mean_val = np.empty([iteration])
min_val = np.empty([iteration])
max_val = np.empty([iteration])

for i in range(iteration):
    fitness(population, layers)
    min_val[i]  = min(population[:, 0])
    mean_val[i] = np.mean(population[:,0])
    max_val[i]  = max(population[:,0])
    print("gen:{:4.0f}\t min:{:5.0f}\t mean:{:5.0f}\t max:{:5.0f}".format(
            i, min_val[i], mean_val[i], max_val[i]))
    population = new_generation(population, 0.4)
    
    fitness(population[-1:], layers, True)
    if i % 20 == 0:
        np.save("population", population)
    