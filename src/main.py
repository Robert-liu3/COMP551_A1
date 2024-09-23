from q1.ITT import fetch_data_ITT, check_data
from q1.CDC import fetch_data_CDC
from q2.LinearRegression import LinearRegression
from q3.Performance import performance

def main():
    print("Hello World!")

if __name__ == "__main__":
    x,y = fetch_data_ITT()
    for i in range(2, 9):
        performance(x, y, size=i*0.1)

