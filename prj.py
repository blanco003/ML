#### CLASSE MULTI LAYER PERCEPTRON

## numero di neuroni di input
## numero di hidden layer e numero di unita
## numero di output
## funzione di attivazione
## learning rate : eta
## regolarizzazione : lambda
## momento : alfa

## inizializzazione
## forward 
## backward
## predict
## grid_search
## loss_function


### CLASSE LAYER

## matrice dei pesi 
## vettore dei bias
## forward 
## backward

import numpy as np

def sigmoid(x):
    return 1/1+np.exp(-x)

def relu(x):
    return np.maximum(0,x)

def dz_relu(x):
    if x>0:
        return 1
    return 0

def dz_sigmoid(x):
    return sigmoid(x)(1-sigmoid(x))


def mse(y_hat, y):
    diff = y_hat - y
    squares = np.square(diff)
    sum = np.sum(squares)
    return 1/len(y_hat) * sum

class Layer:

    # activation function come parametro funzione
    def __init__(self, n_inputs, n_neurons, activation_fun):
        """
        activation_fun : funzione di attivazione del layer
        """

        self.weights = np.random.uniform(low=-0.7, high=0.7, size=(n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))
        self.activation_fun = activation_fun

    def forward(self, input):
        output = np.matmul(input, self.weights) + self.biases      # matmul o dot

        self.output_senza_activation = output  # salvato per backward

        if self.activation_fun=="sigmoid":
            self.output = sigmoid(output)
        elif self.activation_fun=="relu":
            self.output = relu(output)

        print(f"layer output : {self.output}")
        return  self.output  
    

    def backward(self, delta):
        if self.activation_fun == "sigmoid":
            output_delta = delta * dz_sigmoid(self.output_senza_activation)
        else:
            output_delta = np.dot(delta, np.array(self.weights).T) * dz_relu(self.output_senza_activation)
        return output_delta




class MLP:

    def __init__(self, n_inputs, n_outputs, learning_rate):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.layers = []

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    

    def backward(self, y_hat, y):
        delta = y_hat-y
        for layer in self.layers():
            delta = layer.backward(delta)




        

# conotrllo : gli ouput di ogni layer devono coincidere con gli input del prossimo


layer1 = Layer(2,2,"relu")
layer2 = Layer(2,1,"sigmoid")

net = MLP(2,1,1)

net.layers.append(layer1)
net.layers.append(layer2)

input = np.random.uniform(-2,2,size=(1,2))
print(f"input : {input}")

print(f"output : {net.forward(input)}")

