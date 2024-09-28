
import numpy as np
# import matplotlib.pyplot as plt


class SGDLinearRegression:
    def __init__(self, add_bias=False, learning_rate=.001, epsilon=1e-8, max_iters=1e4, batch_size=32, record_history=False):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                        #to get the tolerance for the norm of gradients 
        self.max_iters = max_iters 
        self.batch_size = batch_size                   #maximum number of iteration of gradient descent
        self.record_history = record_history

        if record_history:
            self.w_history = []                 #to store the weight history for visualization
        self.w = None
        pass

    def cost_fn(self, x, y, w): # https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/
        N,D = x.shape                                                       
        z = np.dot(x, w)
        J = .5* np.mean((z - y)**2)
        return J

    def gradient(self, x, y):
        N,D = x.shape
        yh =  x @ self.w 
        grad = np.dot(x.T, (yh - y)) / N
        return grad 

    def mini_batch(self, x, y, batch_size): # # https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/
        mini_batches = []
        data = np.hstack((x, y))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // batch_size
        i = 0
    
        for i in range(n_minibatches + 1):
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
            X_mini = mini_batch[:, :-y.shape[1]]
            Y_mini = mini_batch[:, -y.shape[1]:]
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-y.shape[1]]
            Y_mini = mini_batch[:, -y.shape[1]:]
            mini_batches.append((X_mini, Y_mini))
        return mini_batches    
    
    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])

        N,D = x.shape
        M = y.shape[1]
        self.w = np.zeros((D, M))
        g = np.inf 
        t = 0
        # the code snippet below is for gradient descent
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            mini_batches = self.mini_batch(x, y, self.batch_size)
            for mini_batch in mini_batches:
                x_mini, y_mini = mini_batch
                g = self.gradient(x_mini, y_mini)
                self.w = self.w - self.learning_rate * g 
            
            t += 1
        return self
    
    def predict(self, x):
        N = x.shape[0]   
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        yh = x@self.w
        return yh