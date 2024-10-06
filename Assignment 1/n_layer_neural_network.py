import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

class Layer:
    """
    This class implements a single layer in the deep neural network.
    """
    def __init__(self, input_dim, output_dim, activation_function):
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))
        self.activation_function = activation_function

    def actFun(self, z, type):
        if type.lower() == 'tanh':
            return np.tanh(z)
        if type.lower() == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif type.lower() == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError('Activation function must be tanh, sigmoid, or relu')

    def diff_actFun(self, z, type):
        if type.lower() == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif type.lower() == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif type.lower() == 'relu':
            return np.where(z > 0, 1, 0)
        else:
            raise ValueError('Unsupported activation function.')

    def feedforward(self, A_prev):
        self.Z = A_prev.dot(self.W) + self.b
        self.A = self.actFun(self.Z, type=self.activation_function)
        return self.A

    def backprop(self, A_prev, dA, reg_lambda):
        m = A_prev.shape[0]
        dZ = dA * self.diff_actFun(self.Z, self.activation_function)
        dW = 1 / m * A_prev.T.dot(dZ) + reg_lambda * self.W
        db = 1 / m * np.sum(dZ, axis=0, keepdims=True)
        dA_prev = dZ.dot(self.W.T)
        return dW, db, dA_prev

class DeepNeuralNetwork:
    def __init__(self, layer_dims, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param layer_dims: list containing the dimensions of each layer
        '''
        self.layer_dims = layer_dims
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.num_layers = len(layer_dims)
        np.random.seed(seed)

        # Initialize weights and biases
        self.parameters = {}
        for l in range(1, self.num_layers):
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l - 1], layer_dims[l]) / np.sqrt(layer_dims[l - 1])
            self.parameters['b' + str(l)] = np.zeros((1, layer_dims[l]))

    def actFun(self, z, type):
        if type.lower() == 'tanh':
            return np.tanh(z)
        if type.lower() == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif type.lower() == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError('Activation function must be tanh, sigmoid, or relu')

    def diff_actFun(self, z, type):
        if type.lower() == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif type.lower() == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif type.lower() == 'relu':
            return np.where(z > 0, 1, 0)
        else:
            raise ValueError('Unsupported activation function.')

    def feedforward(self, X):
        A = X
        self.caches = {'A0': A}  # store activations for each layer

        for l in range(1, self.num_layers):
            Z = A.dot(self.parameters['W' + str(l)]) + self.parameters['b' + str(l)]
            A = self.actFun(Z, type=self.actFun_type)
            self.caches['A' + str(l)] = A
            self.caches['Z' + str(l)] = Z

        # Output probabilities using softmax
        exp_scores = np.exp(Z)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X)
        correct_logprobs = -np.log(self.probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs)
        data_loss += self.reg_lambda / 2 * np.sum([np.sum(np.square(self.parameters['W' + str(l)])) for l in range(1, self.num_layers)])
        return (1. / num_examples) * data_loss

    def backprop(self, X, y, learning_rate):
        m = X.shape[0]
        delta = self.probs
        delta[range(m), y] -= 1  # Cross-entropy gradient for softmax

        for l in reversed(range(1, self.num_layers)):
            dW = self.caches['A' + str(l - 1)].T.dot(delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            self.parameters['W' + str(l)] -= learning_rate * dW
            self.parameters['b' + str(l)] -= learning_rate * db

            if l > 1:  # Skip backprop for input layer
                delta = delta.dot(self.parameters['W' + str(l)].T) * self.diff_actFun(self.caches['Z' + str(l - 1)], self.actFun_type)

    def fit_model(self, X, y, learning_rate=0.01, num_epochs=20000, print_loss=True):
        for i in range(num_epochs):
            # Forward pass
            self.feedforward(X)

            # Backward pass
            self.backprop(X, y, learning_rate)

            # Print the loss every 1000 iterations
            if print_loss and i % 1000 == 0:
                print(f"Loss after iteration {i}: {self.calculate_loss(X, y)}")

    def predict(self, X):
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)


def main():
    X, y = generate_data()

    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    layer_dims = [2, 10, 10, 2]
    model = DeepNeuralNetwork(layer_dims, actFun_type='tanh')
    model.fit_model(X, y, learning_rate=0.01)

    plt.title("actFun_type='tanh'")
    plot_decision_boundary(lambda x: model.predict(x), X, y)

if __name__ == "__main__":
    main()
