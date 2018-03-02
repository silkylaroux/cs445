import numpy as np
import mlutilities as ml
import matplotlib.pyplot as plt
from copy import copy
import time
import neuralNetworkA2 as nn

class NeuralNetworkReLU(nn.NeuralNetwork):
    
    def activation(self, weighted_sum):
        return np.maximum(0,weighted_sum,0)