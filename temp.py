# from tensorflow import keras
import keras
from keras import Input
from keras.layers import Dense
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.src.layers import Flatten

from data import Data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model = keras.Sequential([
    Dense(1024, activation="linear"),
    Dense(512, activation="linear"),
    Dense(256, activation="linear"),
    Dense(1, activation="linear"),
])
# print(model.summary())


def det_coeff(y_true, y_pred):
    u = keras.backend.sum(keras.backend.square(y_true - y_pred))
    v = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true)))
    return keras.backend.ones_like(v) - (u / v)


model.compile(optimizer='Adam',
              loss='mae',
              metrics=[det_coeff]
              )
# __________________^^^^ "mae", "mape", "msle" <<< плохо работает стоит попробовать без нормализации
# metrics=['accuracy']


D = Data()

D.create_date_set_1_par(100 * 1, 4, 400_000, 1)

m1 = 400_000 // 10

x = np.array(D.data_set[0:m1 * 8])
y = np.array(D.answer[0:m1 * 8])

x_test = np.array(D.data_set[m1 * 8 + 1:400_000])
y_test = np.array(D.answer[m1 * 8 + 1:400_000])


# y_test_nor = []
#
# for i in range(len(y_test)):
#     y_test_nor.append(float(y_test[i]) * (D.max[3] - D.min[3]) + D.min[3])
#
# plt.plot(y_test_nor)
# plt.grid(True)
# plt.show()


print("Начало обучения")
his = model.fit(x, y, batch_size=512, epochs=5, validation_split=0.2)
print("Конец обучения")
# model.evaluate(x, y)


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
    er.append(abs(y_test[i] - nndata[i]) / y_test[i])

plt.plot(y_test_nor)
plt.plot(nndata_nor)
plt.grid(True)
plt.show()

plt.plot(er)
plt.grid(True)
plt.show()

print(sum(er) / len(er))


# x_test_n = np.array([x_test[0]])
# r = []
# for i in range(1_00):
#     val = model.predict(np.array(x_test_n))[0][0]
#     r.append(val)
#     x_test_n = np.array([np.append(x_test_n[0][1:100], [val])])
#
# plt.plot(y_test)
# plt.plot(r)
# plt.grid(True)
# plt.show()
