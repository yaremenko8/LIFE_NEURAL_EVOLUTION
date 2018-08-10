import numpy as np
import math
import random as rnd
from scipy.special import expit

def randinit(layers): #list of qunatities of neurons in each layer including the bias (from input to output)
    prev = layers[0] #input layer
    weights = []
    for i in layers[1:-1]:
        weights.append(np.array([[rnd.random() * 2 - 1 for k in range(prev)] for j in range(i - 1)] + [[0 for k in range(prev)]]))
        prev = i
    weights.append(np.array([[rnd.random() * 2 - 1 for k in range(prev)] for j in range(layers[-1])]))
    return weights
        
        
        
sigmoid1 = lambda x: np.reciprocal(np.add(1, np.exp(np.negative(x))))
sigmoid2 = np.vectorize(lambda x: 1/(1 + math.exp(-x)))
sigmoid3 = expit

#the last neuron of each layer is bias
#act accepts a vector (u might wanna use np.vectorize)
def fprop(weights, inputs, act, final): #inputs include the bias
    for i in weights[:-1]:
        inputs = act(i.dot(inputs))
        inputs[-1] = 1
    inputs = weights[-1].dot(inputs)
    if final != None:
        return final(inputs)
    return inputs

