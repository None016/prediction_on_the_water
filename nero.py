import keras
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

from data import Data


model = keras.Sequential([
    Dense(1024, activation="linear"),
    Dense(512, activation="linear"),
    Dense(256, activation="linear"),
    Dense(1, activation="linear"),
])

model.compile(optimizer='Adam',
              loss='mae',)
# __________________^^^^ "mae", "mape", "msle" <<< плохо работает стоит попробовать без нормализации
# metrics=['accuracy']

inpex_par = 1

D = Data()

D.create_date_set_1_par(100 * 1, inpex_par, 400_000, 10)

m1 = 400_000 // 10

x = np.array(D.data_set[0:m1 * 8])
y = np.array(D.answer[0:m1 * 8])

x_test = np.array(D.data_set[m1 * 8 + 1:400_000])
y_test = np.array(D.answer[m1 * 8 + 1:400_000])


print("Начало обучения")
his = model.fit(x, y, batch_size=512, epochs=5, validation_split=0.2)
print("Конец обучения")


model.save("model/model0.0.2_tds")

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
    nndata_nor.append(nndata[i] * (D.max[4] - D.min[4]) + D.min[4])
    y_test_nor.append(y_test[i] * (D.max[4] - D.min[4]) + D.min[4])
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
print(mae_test * (D.max[inpex_par] - D.min[inpex_par]))
