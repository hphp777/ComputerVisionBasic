- Purpose of this code
 This code is made to complete loss, training, predict function

- How to run this code
 In the jupyter notebook, add this file in the classifier folder and it will be imported to two_layer_net.ipynb.

- How to adjust parameters (if any)

def __init__(self, input_size, hidden_size, output_size, std=1e-4):
input_size: Give the number of samples
hidden_size: Give the number of hidden layers
output_size: Give the number of output

def loss(self, X, y=None, reg=0.0):
X: Give the input datas
y: Give the lables corresponding to the input data
reg: Give the regularization strength

def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
X: Give the input datas
y: Give the lables corresponding to the input data
X_val: Give the validation dataset
y_val: Give the lables corresponding to the the validation dataset
learning_rate: Give the learning rate(hyper parameter)
learning_rate_decay: Give  scalar giving factor used to decay the learning rate after each epoch.
reg: Give the regularization strength
num_iters: Decide how many you want to iterate
batch_size: Decide the batch size
verbose: if true print progress during optimization.

def predict(self, X):
X: Give the input test data.


- How to define default parameters
def __init__(self, input_size, hidden_size, output_size, std=1e-4):
input_size: Give the number of samples
hidden_size: Give the number of hidden layers
output_size: Give the number of output

def loss(self, X, y=None, reg=0.0):
X: Give the input datas
y: Give the lables corresponding to the input data
reg: Give the regularization strength

def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
X: Give the input datas
y: Give the lables corresponding to the input data
X_val: Give the validation dataset
y_val: Give the lables corresponding to the the validation dataset
learning_rate: Give the learning rate(hyper parameter)
learning_rate_decay: Give  scalar giving factor used to decay the learning rate after each epoch.
reg: Give the regularization strength
num_iters: Decide how many you want to iterate.
batch_size: Decide the batch size. normaly 64.
verbose: If true print progress during optimization. Set True.