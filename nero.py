import keras
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

from data import *


model = keras.Sequential([
    Dense(1024, activation="sigmoid"),
    Dense(512, activation="sigmoid"),
    Dense(256, activation="sigmoid"),
    Dense(1, activation="linear"),
])

model.compile(optimizer='Nadam',
              loss='mean_squared_error',)
# __________________^^^^ "mean_squared_error", "msle" <<< плохо работает стоит попробовать без нормализации
# metrics=['accuracy']

inpex_par = 3
col = 450_000

D = Data()
D_test = Clean_data("data\orp\\3_day.txt")

D.create_date_set_1_par(100 * 1, inpex_par, col, 100)
D_test.normalization(100, 100)

x = np.array(D.data_set[0:col])
y = np.array(D.answer[0:col])

x_test = np.array(D_test.data_set)
y_test = np.array(D_test.answer)


print("Начало обучения")
his = model.fit(x, y, batch_size=32, epochs=15, validation_split=0.2)
print("Конец обучения")


model.save("model/orp_2.1")

nndata = model.predict(x_test)

print(nndata)
print(y_test)

plt.plot(y_test)
plt.plot(nndata)
plt.grid(True)
plt.show()

nndata_nor = []
y_test_nor = []

er = []

for i in range(len(nndata)):
    nndata_nor.append(nndata[i] * (D_test.max - D_test.min) + D_test.min)
    y_test_nor.append(y_test[i] * (D_test.max - D_test.min) + D_test.min)
    er.append(abs(y_test[i] - nndata[i]))

plt.plot(y_test_nor)
plt.plot(nndata_nor)
plt.grid(True)
plt.show()

plt.plot(er)
plt.grid(True)
plt.show()

mae_test = np.sum(er) / len(er)
print(mae_test)
print(mae_test * (D_test.max - D_test.min))
