import numpy as np

class LogisticRegression:  
    def __init__(self, add_bias=True, learning_rate=.1, epsilon=1e-4, max_iters=1e5, batch_size=32, verbose=False):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                         # Tolerance for the norm of gradients 
        self.max_iters = max_iters                     # Maximum number of iteration of gradient descent
        self.batch_size = batch_size                   # Batch size for mini-batch gradient descent
        self.verbose = verbose
    
    def logistic(self, z):
        return 1. / (1 + np.exp(-z)) 

    def gradient(self, x, y):
        N, D = x.shape
        yh = self.logistic(np.dot(x, self.w))          # Predictions size N
        grad = np.dot(x.T, yh - y) / N                 # Gradient for logistic regression
        return grad  
    
    def fit(self, x, y):
        """ Fit model using mini-batch gradient descent """
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x, np.ones(N)])       # Add bias term
        N, D = x.shape
        self.w = np.zeros(D)                           # Initialize weights to zeros
        g = np.inf                                     # Initialize gradient to infinity
        t = 0                                          # Iteration counter
        
        # Mini-batch gradient descent
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            # Shuffle the data at each iteration
            indices = np.random.permutation(N)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            # Loop over batches
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Compute gradient on the batch
                g = self.gradient(x_batch, y_batch)
                self.w = self.w - self.learning_rate * g  # Update weights
            
            t += 1
            
            if self.verbose:
                print(f'Iteration {t}, Gradient norm: {np.linalg.norm(g)}')
        
        if self.verbose:
            print(f'Terminated after {t} iterations, with norm of the gradient equal to {np.linalg.norm(g)}')
            print(f'Final weights: {self.w}')
        return self
    
    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x, np.ones(Nt)])
        yh = self.logistic(np.dot(x, self.w))            # Predict output
        return (yh >= 0.5).astype(int)                   # Return binary predictions (0 or 1)
