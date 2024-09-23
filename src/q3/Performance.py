from q2.LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def performance(x,y,size=0.2):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=size, random_state=42)

    model = LinearRegression(add_bias = False)
    model.fit(X_train,Y_train)
    yh = model.predict(X_test)

    plt.scatter(Y_test, yh)
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('Performance of the model with test size = {}'.format(size))
    plt.show()


