from linearRegression import linear_regression
from buildSchedule import schedule
def main():
    """
    Главная функция, вызывает функцию линейной регрессии и построения графика.
    """
    linear_regression("time_messagees.txt")

if __name__ == "__main__":
    main()
