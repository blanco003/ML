from Layer import Layer
import numpy as np

class MLP:

    def __init__(self, layers_size, learning_rate, activation_hidden, activation_out, epochs):

        #layers_size = [input layer dim, ..., i-th hidden layer dim, ..., output layer dim]

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []

        # input e hidden layer
        for i in range(len(layers_size) - 2):
            self.layers.append(Layer(layers_size[i], layers_size[i+1], activation_hidden))
            print(f"Layer {i} : [{layers_size[i]}, {layers_size[i+1]}]")

        # output layer
        self.layers.append(Layer(layers_size[-2], layers_size[-1], activation_out))
        print(f"Layer output : [{layers_size[-2]}, {layers_size[-1]}]")


    def forward(self, input):

        for layer in self.layers:
            input = layer.forward(input)
        return input
    

    def backward(self, diff):

        #TODO : diff è da passare la prima volta come y_hat - y durante il training
        
        for layer in reversed(self.layers):
            diff = layer.backward(diff)

    def update_parameters(self):

        for layer in self.layers:
            layer.update_parameters(self.learning_rate)

    def fit(self, X, y_hat):

        # TODO: controlla Numpy

        for epoch in range(self.epochs):
            print(f"\n\nEpoch {epoch}")
            print("\nForward")
            o = self.forward(X)
            print(f"o {o.shape}, y_hat {y_hat.shape}")


            print(f"{(y_hat-o).shape}")

            print("\nBackward")
            self.backward(y_hat-o)

            print("\nUpdate parameters")
            self.update_parameters()


# TODO controllo : gli ouput di ogni layer devono coincidere con gli input del prossimo
