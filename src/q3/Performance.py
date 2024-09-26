from q2.LinearRegression import LinearRegression
from q2.LogisticRegression import LogisticRegression
from q1.ITT import fetch_data_ITT
from q1.CDC import fetch_data_CDC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def performanceLinearRegression(size=0.2):
    x,y = fetch_data_ITT()
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=size, random_state=42)

    model = LinearRegression(add_bias = False)
    model.fit(X_train,Y_train)
    yh = model.predict(X_test)

    plt.scatter(Y_test, yh)
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('Performance of the model with test size = {}'.format(size))
    plt.show()

def performanceLogisticRegression(size = 0.2):
    x,y = fetch_data_CDC()
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=size, random_state=42)

    model = LogisticRegression(verbose=True)
    model.fit(X_train,Y_train)
    yh = model.predict(X_test)

    yh_binary = (yh > 0.5).astype(int)

    # Compute the metrics
    accuracy = accuracy_score(Y_test, yh_binary)
    precision = precision_score(Y_test, yh_binary)
    recall = recall_score(Y_test, yh_binary)
    f1 = f1_score(Y_test, yh_binary)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

