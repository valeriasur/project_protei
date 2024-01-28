import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from convertingHoursToSeconds import converting_hours_to_seconds
from standardizationValues import standardization_values
from buildSchedule import schedule

def linear_regression(file_path):
    """
    Выполняет линейную регрессию на основе данных времени и количества сообщений,
    выводит метрики качества для оценки качества предсказаний модели линейной регрессии.
    Параметры:
    file_path (str) : путь файла
    """
    Dataframe = pd.read_csv(file_path, delimiter=",", header=None)
    Dataframe = Dataframe.rename(columns={0: "time", 1: "number_of_messages"})
    # смена типа данных number_of_messages на int, так как количество сообщений не может быть дробным числом
    Dataframe["number_of_messages"] = Dataframe["number_of_messages"].astype(int)
    # вызов функции, которая преобразует строковое значение времени в общее количество секунд
    Dataframe['total_seconds'] = Dataframe['time'].apply(converting_hours_to_seconds)
    Dataframe = Dataframe.reindex(columns=['total_seconds', 'number_of_messages'])
    x = Dataframe['total_seconds']
    y = Dataframe['number_of_messages']
    # вызов функции, которая стандартизирует данные, разделяет данные на обучающий и проверочный наборы 
    x_scaled, y_scaled, x_train, y_train, x_valid, y_valid = standardization_values(x,y)
    Model = LinearRegression()
    Model.fit(x_train, y_train)
    predictions_valid = Model.predict(x_valid)
    # вызов функции, которая строит график 
    schedule(x_scaled, y_scaled, x_valid, predictions_valid)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_valid, predictions_valid))
    print('Mean Squared Error:', metrics.mean_squared_error(y_valid, predictions_valid))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_valid, predictions_valid)))
    print('R2 metric:', metrics.r2_score(y_valid, predictions_valid))
    
