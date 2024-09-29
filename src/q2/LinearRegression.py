import numpy as np
# import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        self.w = None
        pass
    
    def fit(self, x, y):
            if x.ndim == 1:
                x = x[:, None] 
            N = x.shape[0]
            if self.add_bias:
                x = np.column_stack([x,np.ones(N)])   
            self.w = np.matmul(
                np.linalg.inv(np.matmul(x.T,x))
                ,np.matmul(x.T,y)
                )
            print("this is the shape of x after fit: ", x.shape)
            print("this is the shape of w after fit: ", self.w.shape)         
            return self
        
    def predict(self, x):
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        print("this is the shape of x before predict: ", x.shape)
        print("this is the shape of w before predict: ", self.w.shape)
        yh = np.matmul(x,self.w)
        return yh