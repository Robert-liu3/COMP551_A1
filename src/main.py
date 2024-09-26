from q1.ITT import fetch_data_ITT, check_data
from q1.CDC import fetch_data_CDC
from q2.LinearRegression import LinearRegression
from q2.LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from q3.Performance import performance

def main():
    print("Hello World!")

if __name__ == "__main__":
    # x,y = fetch_data_ITT()
    # for i in range(2, 9):
    #     performance(x, y, size=i*0.1)
    
    N = 50
    x = np.linspace(-5,5, N)
    y = ( x < 2).astype(int)                                  #generate synthetic data
    model = LogisticRegression(verbose=True, )
    yh = model.fit(x,y).predict(x)
    plt.plot(x, y, '.', label='dataset')
    plt.plot(x, yh, 'g', alpha=.5, label='predictions')
    plt.xlabel('x')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()
