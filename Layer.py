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
        self.unactivated_output = np.dot(self.inputs, self.weights) + self.bias     # salvato output senza funzione di attivazione
        print(f"self.unactivated_output : {self.unactivated_output.shape} = {self.inputs.shape} * {self.weights.shape} + {self.bias.shape}")
        output = self.activation.function(self.unactivated_output)   # non serve salvare l'output dopo la funzione di attivazione
        
        #print(f"layer output : {output}")
        return  output  
    

    def backward(self, diff):

        self.delta = self.activation.derivative(self.unactivated_output) *  diff 
        print(f"self.delta : {self.delta.shape} = {diff.shape} * {self.activation.derivative(self.unactivated_output).shape}")
       
        err = np.dot(self.delta, np.transpose(self.weights)) 

        print(f"err {err.shape} = {self.delta.shape} * {np.transpose(self.weights).shape}")
        
        return  err
    

    def update_parameters(self, learning_rate):

        delta_W = np.dot(np.transpose(self.inputs), self.delta) * (1/self.inputs.shape[0])
        print(f"delta_W {delta_W.shape} = {np.transpose(self.inputs).shape} * {self.delta.shape} * 1/{self.inputs.shape[0]}")
        
        #delta_b = (1/self.inputs.shape[0]) * self.delta
        delta_b = np.mean(self.delta, axis=0, keepdims=True)
        print(f"delta_b {delta_b.shape} = 1/{self.inputs.shape[0]} * {self.delta.shape}")

        self.weights = self.weights + learning_rate * delta_W
        self.bias = self.bias + learning_rate * delta_b
