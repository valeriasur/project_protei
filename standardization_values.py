from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np


def standardization_values(x, y):
    """
    Выполняет стандартизацию числовых значений.
    Параметры:
    x (pandas.Series) : столбцец датафрейма Dataframe, который содержит данные столбца total_seconds.
    y (pandas.Series) : столбцец датафрейма Dataframe, содержит данные столбца number_of_messages.
    Возвращаемое значение:
    x_scaled (np.array) : Массив numpy, содержащий стандартизованные значения столбца total_seconds.
    y_scaled (np.array) : Массив numpy, содержащий стандартизованные значения столбца number_of_messages.
    x_train (np.array) : Массив numpy, содержащий стандартизованные значения столбца total_seconds для обучающего набора данных.
    y_train (np.array) : Массив numpy, содержащий стандартизованные значения столбца number_of_messages для обучающего набора данных.
    x_valid (np.array) : Массив numpy, содержащий стандартизованные значения столбца total_seconds для валидационного набора данных.
    y_valid (np.array) : Массив numpy, содержащий стандартизованные значения столбца number_of_messages для валидационного набора данных.
    """
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(np.array(x).reshape(-1, 1))
    y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_scaled, y_scaled, test_size=0.25, random_state=0)
    return x_scaled, y_scaled, x_train, y_train, x_valid, y_valid
