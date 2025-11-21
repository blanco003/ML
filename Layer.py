import numpy as np
from Function import Function


class Layer:

    def __init__(self, n_inputs, n_neurons, activation : Function):
        """
        activation_fun : funzione di attivazione del layer
        """

        self.weights = np.random.uniform(low=-0.7, high=0.7, size=(n_inputs, n_neurons))
        self.bias = np.zeros((1, n_neurons))
        self.activation = activation


    def forward(self, inputs):

        self.inputs = inputs  # salvataggio input del layer : utile per backprop e aggiornamento dei pesi
        self.unactivated_output = np.matmul(self.inputs, self.weights) + self.bias     # salvato output senza funzione di attivazione
        output = self.activation.function(self.unactivated_output)   # non serve salvare l'output dopo la funzione di attivazione
        
        print(f"layer output : {output}")
        return  output  
    

    def backward(self, diff):

        self.delta = diff * self.activation.derivative(self.unactivated_output)
        
        return np.dot(self.delta, np.transpose(self.weights)) # error