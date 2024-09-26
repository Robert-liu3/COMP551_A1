from q1.ITT import fetch_data_ITT, check_data
from q1.CDC import fetch_data_CDC
from q2.LinearRegression import LinearRegression
from q2.LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from q3.Performance import performanceLinearRegression, performanceLogisticRegression

def main():
    print("Hello World!")

if __name__ == "__main__":
    # x,y = fetch_data_ITT()
    # for i in range(2, 9):
    #     performance(x, y, size=i*0.1)
    
    x,y = fetch_data_CDC()                            #generate synthetic data

    model = LogisticRegression(verbose=True, )
    print("this is the shape of x: ", x.shape)
    print("this is the shape of y: ", y.shape)

    performanceLogisticRegression(x, y, size=0.2)
    # yh = model.fit(x,y).predict(x)
    # plt.plot(x, y, '.', label='dataset')
    # plt.plot(x, yh, 'g', alpha=.5, label='predictions')
    # plt.xlabel('x')
    # plt.ylabel(r'$y$')
    # plt.legend()
    # plt.show()
