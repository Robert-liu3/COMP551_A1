from q1.ITT import fetch_data_ITT
from q2.LinearRegression import LinearRegression

def main():
    print("Hello World!")

if __name__ == "__main__":
    model = LinearRegression()
    main()
    x,y = fetch_data_ITT()
    model.fit(x,y)

