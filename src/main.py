from q1.ITT import fetch_data_ITT, check_data
from q1.CDC import fetch_data_CDC
from q2.LinearRegression import LinearRegression
from q2.SGDLinearRegression import SGDLinearRegression
from q2.LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt

def main():
    print("Hello World!")

if __name__ == "__main__":
    model = LogisticRegression()
    # print("Hello World!")
    # main()
    x,y = fetch_data_CDC()
    # check_data(x,y)
    yh = model.fit(x,y).predict(x)
    plt.plot(x, y, '.')
    plt.plot(x, yh, 'g-', alpha=.5)
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.xlim(-20,20)
    #plt.ylim(-100,100)
    plt.show()
    #print("this is the weights", model.w)
    # fetch_data_CDC()

