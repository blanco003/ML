import math
import numpy as np

class Function:

    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.derivative = derivative
        

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return np.multiply(sigmoid(x), (1-sigmoid(x)))

def relu(x):
    return np.maximum(0,x)

def relu_deriv(x):
    return (x > 0).astype(float)
    

Sigmoid = Function('Sigmoid', sigmoid, sigmoid_deriv)
ReLU = Function('ReLU', relu, relu_deriv)


act_funcs = {
    'Sigmoid': Sigmoid,
    'ReLU': ReLU
}

def mse(y_hat, y):
    return 1/len(y_hat) * np.sum(np.square(y_hat - y))

