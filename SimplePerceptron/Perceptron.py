import numpy as np

def step(z):
    return 1 if (z > 0) else 0

class Perceptron:
    def __init__(self, alpha=0.001, max_iter=1000, activation=step):
        self.alpha = alpha
        self.max_iter = max_iter
        self.activation = activation
        self.misses = []
        self.weights = None

    def fit(self, X, y):
        #m: Samples. n: Features.
        m, n = X.shape

        #Initiallize weights
        w = np.zeros((n+1, 1))

        #Recall misses to show as %
        misses = []

        #Train
        iter = 0
        while iter < self.max_iter:
            missed = 0

            for index, x in enumerate(X):
                #Place bias term for each x
                x = np.insert(x, 0 , 1).reshape(-1, 1)
                
                #Evaluate activation function: Dot product of parameters with weights
                y_hat = self.activation(np.dot(x.T, w))
                
                #Update if missclassified
                if y_hat - y[index] != 0:
                    w += self.alpha * ((y[index] - y_hat) * x)
                    missed += 1
            
            misses.append(missed)
            iter += 1
        
        self.misses = misses
        self.weights = w





