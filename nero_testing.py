import keras
from keras import Input
from keras.layers import Dense
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.src.layers import Flatten

from data import Data


def det_coeff(y_true, y_pred):
    u = keras.backend.sum(keras.backend.square(y_true - y_pred))
    v = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true)))
    return keras.backend.ones_like(v) - (u / v)


model = keras.models.load_model("model/model0.0.2_tds")

print(model.summary())


D = Data()
D.create_date_set_1_par(100 * 1, 4, 400_000, 30)

m1 = 400_000 // 10

x = np.array(D.data_set[0:m1 * 8])
y = np.array(D.answer[0:m1 * 8])

x_test = np.array(D.data_set[m1 * 8 + 1:400_000])
y_test = np.array(D.answer[m1 * 8 + 1:400_000])

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



