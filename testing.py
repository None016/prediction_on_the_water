import keras
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

from data import *

model = keras.models.load_model("model/orp_2.0")

D_test = Clean_data("data\orp\\3_day.txt")
D_test.normalization(100, 100)

print(D_test.min)
print(D_test.max)

x_test = np.array(D_test.data_set)
y_test = np.array(D_test.answer)

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


