import numpy as np
from typing import List

class NeuralNetwork:
    def __init__(self, layers:List[int], genes:np.ndarray):        
        self.NN = []
        self.genes = genes
        end = 0

        for i, _ in enumerate(layers[:-1]):
            start = end
            end += layers[i]*layers[i+1]
            if end > genes.shape[0]:
                print("GENES LIMIT EXCEEDED")
                # Just stopping is not the best way to handle these situation,
                # but being a self-project I'll be fine with just a warning
                break
            connections = np.reshape( genes[start:end], (layers[i+1], layers[i]) )
            self.NN.append(connections)
    
    def getGenes(self)->np.ndarray:
        return self.genes
    
    def sigmoid(self, x:float)->float:
        return 1/(1+np.exp(-1*x))
  
    def reLU(self, x:float)->float:
        return x*(x>0)

    def printNN(self)->None:
        print("+-------------------------------+")
        for layer in self.NN:
            print( layer )
            print( layer.shape )
        print("+-------------------------------+")

    
    def forward_propagation(self, x:np.ndarray)->np.ndarray:
        for layer in self.NN[:-1]:
            Ax = layer.dot(x)
            x = self.reLU(Ax)
        # Last layer
        Ax = self.NN[-1].dot(x)
        x = np.tanh(Ax)
        return x
    