# Импортируются нужные библиотеки
import joblib as jb
import numpy as np
import pandas as pd
import sklearn as sns

# Чтение сохраненной модели
xgb_model = jb.load('xgb_model.pkl')

# Вводимые данные
distance = float(input("Введите дистанцию: "))
consume = float(input("Введите расход (Л/100км): "))
speed = int(input("Введите среднюю скорость: "))
temp_inside = float(input("Введите температуру внутри: "))
temp_outside = float(input("Введите температуру снаружи: "))
AC = input("Работал ли кондиционер (Y or N): ")
rain = input("Шел ли дождь (Y or N): ")
sun = input("Было ли солнечно (Y or N): ")

# Обработка вводимых данных в целочисленное значение
if AC == "y" or AC == "Y": # AC
    AC = 1
else:
    AC = 0
# rain
if rain == "y" or rain == "Y": # rain
    rain = 1
else:
    rain = 0

if sun == "y" or sun == "Y": # sun
    sun = 1
else:
    sun = 0

# Запись в список
params_to_pred = []

params_to_pred.append(distance)
params_to_pred.append(consume)
params_to_pred.append(speed)
params_to_pred.append(temp_inside)
params_to_pred.append(temp_outside)
params_to_pred.append(AC)
params_to_pred.append(rain)
params_to_pred.append(sun)

# Зоздание массива для для ввода в модель для получения прогноза
to = np.array([params_to_pred])
to_pred = pd.DataFrame(to, columns=["distance", "consume", "speed", "temp_inside", "temp_outside", "AC", "rain", "sun"])

# Приведение некоторых столбцов к нужному типу
to_pred["speed"] = to_pred["speed"].astype("int64")
to_pred["AC"] = to_pred["AC"].astype("int64")
to_pred["rain"] = to_pred["rain"].astype("int64")
to_pred["sun"] = to_pred["sun"].astype("int64")

# Запись данных в модель
xgb_pred = xgb_model.predict(to_pred)

# Вывод результата прогнозирвания
if xgb_pred == 1:
    print("Результат прогнозирования: по веденным парамтрам определен тип бензина - SP98")
elif xgb_pred == 0:
    print("Результат прогнозирования: по веденным парамтрам определен тип бензина - E10")
