from q2.LinearRegression import LinearRegression
from q2.LogisticRegression import LogisticRegression
from q2.SGDLinearRegression import SGDLinearRegression
from q2.SGDLogisticRegression import SGDLogisticRegression
from q1.ITT import fetch_data_ITT
from q1.CDC import fetch_data_CDC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt


#PERFORMANCE FUNCTIONS FOR LINEAR REGRESSION AND LOGISTIC REGRESSION
def performanceLinearRegression(size=0.2):
    x,y = fetch_data_ITT()
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=size, random_state=42)

    model = LinearRegression(add_bias = False)
    
    model.fit(X_train.values,Y_train.values)
    yh = model.predict(X_test.values)
    Y_test = Y_test.values.ravel()
    yh = yh.ravel()

    mse = mean_squared_error(Y_test, yh)

    weights_df_1 = pd.DataFrame(model.w[:,0], columns=['Weights'])
    weights_df_2 = pd.DataFrame(model.w[:,1], columns=['Weights2'])
    weights_df = pd.concat([weights_df_1, weights_df_2], axis=1)
    weights_df['Size'] = size
    weights_df['Mean Square Error'] = mse

    weights_df.to_csv(f'./q3/csv/itt_weights_{size}.csv', index=False)


    m, b = np.polyfit(Y_test, yh, 1)
    print("Dataset with size %.2f has slope %.4f" % (size,m))

    plt.plot(Y_test, m*Y_test + b, color='red')

    plt.scatter(Y_test, yh)
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('Performance of the model with test size = {}'.format(size))
    plt.show()

    return size, mse

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

    weights_df = pd.DataFrame(model.w, columns=['Weights'])
    weights_df['Size'] = size
    weights_df['Accuracy'] = accuracy
    weights_df['Precision'] = precision
    weights_df['Recall'] = recall
    weights_df['F1'] = f1

    weights_df.to_csv(f'./q3/csv/cdc_weights_{size}.csv', index=False)

    return size, accuracy, precision, recall, f1


#PERFORMANCE FUNCTIONS FOR SGD LINEAR REGRESSION AND SGD LOGISTIC REGRESSION
def performanceSGDLinReg(batch_size=32, size=0.2):
    x,y = fetch_data_ITT()
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=size, random_state=42)
    model = SGDLinearRegression(batch_size=batch_size, learning_rate=0.0000000000000001) #YOU NEED TO TEST OUT THE DIFFERENT LEARNING RATES TO KNOW WHICH ONE IS GOOD
    model.fit(X_train,Y_train)
    yh = model.predict(X_test)
    yh = yh.fillna(yh.mean())
    Y_test = Y_test.values.ravel()
    yh = yh.values.ravel()

    m, b = np.polyfit(Y_test, yh, 1)

    plt.plot(Y_test, m*Y_test + b, color='red')

    plt.scatter(Y_test, yh)
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('Performance of the model with test size = {}'.format(size))
    plt.show()

    mse = mean_squared_error(Y_test, yh)

    return batch_size, mse

def performanceSGDLogReg(batch_size=32, size=0.2):
    x,y = fetch_data_CDC()
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=size, random_state=42)

    model = SGDLogisticRegression(batch_size=batch_size, learning_rate=0.1)
    model.fit(X_train,Y_train)
    yh = model.predict(X_test)
    yh_binary = (yh > 0.5).astype(int)

    accuracy = accuracy_score(Y_test, yh_binary)
    precision = precision_score(Y_test, yh_binary)
    recall = recall_score(Y_test, yh_binary)
    f1 = f1_score(Y_test, yh_binary)

    return batch_size, accuracy, precision, recall, f1

#GROWING SIZES
def linRegGrowingSubset():
    list = []
    for i in range(2, 9):
        list.append(performanceLinearRegression(size=i*0.1))
    df = pd.DataFrame(list, columns=['Size', 'Mean Square Error'])
    plt.scatter(df['Size'], df['Mean Square Error'])
    plt.xlabel('Size')
    plt.ylabel('Mean Square Error')
    plt.title('Mean Square Error vs Size')
    plt.show()

def logRegGrowingSubset():
    list = []
    for i in range(2, 9):
        list.append(performanceLogisticRegression(size=i*0.1))
    df = pd.DataFrame(list, columns=['Size', 'Accuracy', 'Precision', 'Recall', 'F1'])
    plt.scatter(df['Size'], df['Accuracy'], color='red', label='Accuracy')
    plt.scatter(df['Size'], df['Precision'], color='blue', label='Precision')
    plt.scatter(df['Size'], df['Recall'], color='green', label='Recall')
    plt.scatter(df['Size'], df['F1'], color='yellow', label='F1')
    plt.xlabel('Size')
    plt.ylabel('Metrics')
    plt.title('Metrics vs Size')
    plt.legend()
    plt.show()

#GROWING MINI BATCH SIZES
def SGDLinRegGrowingMiniBatch():
    list = []
    for i in range(5, 8):
        list.append(performanceSGDLinReg(batch_size=2**i))
    df = pd.DataFrame(list, columns=['Batch Size', 'Mean Square Error'])
    plt.scatter(df['Batch Size'], df['Mean Square Error'])
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Square Error')
    plt.title('Mean Square Error vs Batch Size')
    plt.show()

def SGDLogRegGrowingMiniBatch():
    list = []
    for i in range(3, 8):
        list.append(performanceSGDLogReg(batch_size=2**i))
    df = pd.DataFrame(list, columns=['Batch Size', 'Accuracy', 'Precision', 'Recall', 'F1'])
    plt.scatter(df['Batch Size'], df['Accuracy'], color='red', label='Accuracy')
    plt.scatter(df['Batch Size'], df['Precision'], color='blue', label='Precision')
    plt.scatter(df['Batch Size'], df['Recall'], color='green', label='Recall')
    plt.scatter(df['Batch Size'], df['F1'], color='yellow', label='F1')
    plt.xlabel('Batch Size')
    plt.ylabel('Metrics')
    plt.title('Metrics vs Batch Size')
    plt.legend()
    plt.show()