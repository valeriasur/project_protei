
def converting_hours_to_seconds(time):
    """
    Преобразует строковое значение времени в общее количество секунд.
    Параметры:
    time (str) : столбцец датафрейма Dataframe, который содержит время, за которое поступает определенное количество сообщений.
    Возвращаемое значение:
    total_seconds (int) : столбец, который содержит общее количество секунд, полученное из часов, минут и секунд.
    """
    time_number = time.split(":")  # разделение строки
    hours = int(time_number[0])  # преобразование каждой части строки в int
    minutes = int(time_number[1])
    seconds = int(time_number[2])
    total_seconds = hours * 3600 + minutes * 60 + \
        seconds  # получение общего кол-ва секунд
    return total_seconds
