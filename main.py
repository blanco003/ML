
from Layer import *
from MLP import *
from Function import *

mlp = MLP(
    layers_size=[2, 5, 5, 1],
    learning_rate=0.01,
    activation_hidden=act_funcs["ReLU"],
    activation_out=act_funcs["Sigmoid"]
)

input = np.random.uniform(-2,2,size=(1,2))

print(f"input : {input}")

print(f"output : {mlp.forward(input)}")

