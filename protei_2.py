import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def linear_regression(file_path):
    """
    Выполняет линейную регрессию на основе данных времени и количества сообщений,
    выводит метрики качества для оценки качества предсказаний модели линейной регрессии.
    Параметры:
    file_path (str) :  путь файла
    """
    global x_scaled, y_scaled, x_valid, predictions_valid
    Dataframe = pd.read_csv(file_path, delimiter=",", header=None)
    Dataframe = Dataframe.rename(columns={0: "time", 1: "number_of_messages"})
    # преобразование столбца time в столбец total_seconds с количеством секунд и типом данных int 
    first_column = Dataframe['time']
    time_number = first_column.str.split(":")
    hours = time_number.apply(lambda x: int(x[0]))
    minutes = time_number.apply(lambda x: int(x[1]))
    seconds = time_number.apply(lambda x: int(x[2]))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    Dataframe['total_seconds'] = total_seconds
    # смена типа данных number_of_messages, тк количество сообщений не может быть дробным числом
    Dataframe["number_of_messages"] = Dataframe["number_of_messages"].astype(int) 
    # создание двух массивов
    x = Dataframe['total_seconds'] 
    y = Dataframe['number_of_messages']
    # стандартизация числовых значений
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(np.array(x).reshape(-1,1)) 
    y_scaled = scaler.fit_transform(np.array(y).reshape(-1,1))
    # разделение данных на обучающий и проверочный наборы 
    x_train,x_valid,y_train,y_valid = train_test_split(x_scaled,y_scaled, test_size=0.25, random_state=0)
    Model = LinearRegression() 
    Model.fit(x_train, y_train) 
    # выполнение предсказания
    predictions_valid = Model.predict(x_valid) 
    # вывод метрик качества для оценки качества предсказаний модели линейной регрессии
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_valid, predictions_valid))
    print('Mean Squared Error:', metrics.mean_squared_error(y_valid, predictions_valid))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_valid, predictions_valid)))
    print('R2 metric:', metrics.r2_score(y_valid, predictions_valid))

def schedule():
    """
    Строит график для столбца 'total_seconds'.
    """
    plt.figure(figsize=(10, 6)) 
    # создание точечного графика
    plt.scatter(x_scaled, y_scaled, c='g', s = 5, label = 'Данные') 
    # создание линейного графика
    plt.plot(x_valid,predictions_valid,'black', label = 'Линейная регрессия') 
    plt.xlabel('Время (независимая переменная)')
    plt.ylabel('Количество сообщений (целевая переменная)')
    plt.legend()
    plt.title('График предсказанных и истинных значений')
    plt.grid()
    plt.show()

def main():
    """
    Главная функция, вызывает функцию линейной регрессии и построения графика.
    """
    linear_regression("time_messagees.txt")
    schedule()

main()