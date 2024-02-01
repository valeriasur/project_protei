# Задание 2
Код программы взят из 1 задания и разбит на модули. Программа по имеющимся данным из лога от 
мобильного оператора строит модель линейной регрессии, затем график данных, а после выводит метрики качества  для оценки эффективности модели. 
# Технологии
- Python
# Использование
Для запуска программы необходимо загрузить все модули, список внешних зависимостей проекта в файле requirements.txt, а также файл time_messagees.txt, который будет считывать программа и который содержит в себе данные о количестве сообщений за какое - то определенное время.\
Установите необходимые внешние зависимости проекта с помощью команды:
```
pip install -r requirements.txt
```
# Интерпретация графика
![изображение](https://github.com/valeriasur/project_protei/assets/103844758/63f43e6e-567b-4031-950d-5f6c3fb5f979)\
Модель хорошо справляется с предсказыванием значений, на нее не влияют выбросы, подъемы и спады количества сообщений. По графику видно, что количество сообщений линейно возрастает со временем.
# Интерпретация значений метрик качества
- Значения метрик MAE, MSE и RMSE составляют соответственно 0.05, 0.008 и 0.08. Значение R2 равно 0.004. Учитывая эти метрики, можно сделать вывод, что модель линейной регрессии имеет относительно низкую погрешность предсказания. 
- Значение коэффициента детерминации (R2) очень низкое, что указывает на то, что модель не объясняет значительную часть дисперсии в данных.
В целом, необходимо провести дополнительный анализ и улучшение модели для достижения более точных предсказаний.
