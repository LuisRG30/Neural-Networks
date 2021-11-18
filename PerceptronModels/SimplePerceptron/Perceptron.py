import numpy as np
import math as m

def step(z):
    return 1 if (z > 0) else 0

def relu(z):
    return min(z, 1) if (z > 0.5) else 0

def sigmoid(z):
    s = 1 / (1 + m.exp(-z + 0.5))
    return min(s, 1) if (s > 0.5) else 0

def tanh(z):
    z -= 0.5
    return round((m.exp(z) - m.exp(-z))/(m.exp(z) + m.exp(-z)))


functions = {
    'step': step,
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh,
}


class Perceptron:
    def __init__(self, alpha=0.001, max_iter=1000, activation='step'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.activation = functions[activation]
        self.misses = []
        self.weights = None

    def fit(self, X, y, mini=False, batch_size=0.05):
        #m: Samples. n: Features.
        m, n = X.shape

        #Initiallize weights
        w = np.zeros((n+1, 1))

        #Recall misses to show as %
        misses = []

        #Train
        iter = 0
        while iter < self.max_iter:
            loss = 0
            if mini:
                pass

            else:
                for index, x in enumerate(X):
                    #Place bias term for each x
                    x = np.insert(x, 0 , 1).reshape(-1, 1)
                    
                    #Evaluate activation function: Dot product of parameters with weights
                    y_hat = self.activation(np.dot(x.T, w))
                    
                    #Update if missclassified
                    if y_hat - y[index] != 0:
                        w += self.alpha * ((y[index] - y_hat) * x)
                        loss += 1
            
            misses.append(loss)
            iter += 1
        
        self.misses = misses
        self.weights = w

    def predict(self, X):
        #Returns some y
        y_hat = []
        for x in X:
            x = np.insert(x, 0 , 1).reshape(-1, 1)
            y_hat.append(self.activation(np.dot(x.T, self.weights)))
        return y_hat

    def score(self, X_test, y_test):
        N = len(X_test)
        if N != len(y_test):
            raise Exception("Inputs X and y have different dimensions.")

        n = 0
        for index, x in enumerate(X_test):
            x = np.insert(x, 0 , 1).reshape(-1, 1)
            if self.activation(np.dot(x.T, self.weights)) != y_test[index]:
                n += 1
            
        return n / N
        




