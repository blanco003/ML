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

from Layer import Layer
import numpy as np

class MLP:

    def __init__(self, layers_size, learning_rate, activation_hidden, activation_out):

        #layers_size = [input layer dim, ..., i-th hidden layer dim, ..., output layer dim]

        self.learning_rate = learning_rate
        self.layers = []

        # input e hidden layer
        for i in range(len(layers_size) - 2):
            self.layers.append(Layer(layers_size[i], layers_size[i+1], activation_hidden))

        # output layer
        self.layers.append(Layer(layers_size[-2], layers_size[-1], activation_out))


    def forward(self, input):

        for layer in self.layers:
            input = layer.forward(input)
        return input
    

    def backward(self, diff):

        #TODO : diff è da passare la prima volta come y_hat - y durante il training
        
        for layer in reversed(self.layers()):
            diff = layer.backward(diff)

        # TODO : update weights




        

# controllo : gli ouput di ogni layer devono coincidere con gli input del prossimo
