
from Layer import *
from MLP import *
from Function import *
def load_monks_file(path):
    X = []
    y = []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 8:
                continue  

            label = int(parts[0])
            features = list(map(int, parts[1:7]))

            y.append(label)
            X.append(features)

    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32).reshape(-1, 1)  # 2D

    return X, y

# Carica train e test
X_train, y_train = load_monks_file("data/monks-1.train")
X_test, y_test  = load_monks_file("data/monks-1.test")


mlp = MLP(
    layers_size=[6, 5, 1],
    learning_rate=0.05,
    activation_hidden=act_funcs["ReLU"],
    activation_out=act_funcs["Sigmoid"], 
    epochs=100
)


print(f"X train : {X_train.shape}")
print(f"y train : {y_train.shape}")

mlp.fit(X_train, y_train)
