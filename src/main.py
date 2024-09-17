from q1.ITT import fetch_data_ITT, check_data
from q1.CDC import fetch_data_CDC


def main():
    print("Hello World!")

if __name__ == "__main__":
    # model = LinearRegression()
    print("Hello World!")
    main()
    x,y = fetch_data_ITT()
    check_data(x,y)
    # model.fit(x,y)
    # fetch_data_CDC()

