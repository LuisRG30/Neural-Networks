import numpy as np

class Perceptron:
    def __init__(self, alpha=0.001, tolerance=0.001, max_iter=1000):
        self.alpha = alpha

    def fit(self, X, y):
        #m: Samples. n: Features.
        m, n = X.shape

        #Initiallize weights
        w = np.zeros((n+1, 1))

        #Recall misses to show as %
        misses = []

        #Train
        iter = 0
        diff = float('inf')
        prev_loss = 0
        while iter < self.max_iter and diff > self.tolerance:
            missed = 0

            for index, x in enumerate(X):
                pass

            iter += 1
