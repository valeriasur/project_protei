import matplotlib.pyplot as plt
def schedule(x_scaled, y_scaled,x_valid,predictions_valid):
    """
    Строит график для столбца 'total_seconds'.
    Параметры:
    x_scaled (np.array) : Массив numpy, содержащий стандартизованные значения столбца total_seconds.
    y_scaled (np.array) : Массив numpy, содержащий стандартизованные значения столбца number_of_messages.
    x_valid (np.array) : Массив numpy, содержащий стандартизованные значения столбца total_seconds для валидационного набора данных.
    predictions_valid (np.array) : Массив numpy, который содержит предсказанные значения модели линейной регрессии для валидационного набора данных.
    """ 
    plt.figure(figsize=(10, 6))
    plt.scatter(x_scaled, y_scaled, c='g', s = 5, label = 'Данные')
    plt.plot(x_valid,predictions_valid,'black', label = 'Линейная регрессия')
    plt.xlabel('Время (независимая переменная)')
    plt.ylabel('Количество сообщений (целевая переменная)')
    plt.legend()
    plt.title('График предсказанных и истинных значений')
    plt.grid()
    plt.show()