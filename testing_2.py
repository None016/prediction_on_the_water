import keras
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

import tests
from data import *

model = keras.models.load_model("model/orp_1000_100_2")

ti = "1_w"

D_test = Clean_data("D:\Pyton\prediction_on_the_water\data\orp\\1_hour.txt")

# D_test.min = 192.2
# D_test.max = 204.8

# D_test.smoothing(100)

D_test.normalization(1000, 1, 100)

print("min: ", D_test.min)
print("max: ", D_test.max)
print("len: ", D_test.len)

x_test = np.array(D_test.data_set)
y_test = np.array(D_test.answer)

nndata = model.predict(x_test)

# y_test = y_test * (D_test.max - D_test.min) + D_test.min
# nndata = nndata * (D_test.max - D_test.min) + D_test.min
#
# summa = 0
# summa2 = 0
# for i in range(len(nndata)):
#     summa2 = 0
#     for j in range(len(nndata[0])):
#         summa2 += (abs(y_test[i][j] - nndata[i][j]) / abs(y_test[i][j]))
#
#     summa += summa2 / len(nndata[0])
#
# print((summa / len(nndata)) * 100)


plt.plot(y_test[1000])
plt.plot(nndata[1000])
plt.grid(True)
plt.savefig(f"svg/test_data({ti}).svg", format='svg', bbox_inches='tight')
plt.show()


mono = []
mono_y = []
for i in range(0, len(y_test), len(y_test[0])):
    mono = [*mono, *nndata[i]]
    mono_y = [*mono_y, *y_test[i]]

plt.plot(mono_y)
plt.plot(mono)
plt.grid(True)
plt.savefig(f"svg/monolit({ti}).svg", format='svg', bbox_inches='tight')
plt.show()

er = []
er_norm = []

for i in range(len(nndata)):
    e_t = []
    for j in range(len(nndata[0])):
        e_t.append(abs(y_test[i][j] - nndata[i][j]))
    er.append(sum(e_t) / len(e_t))
    er_norm.append((sum(e_t) / len(e_t)) * (D_test.max - D_test.min))


plt.plot(er)
plt.grid(True)
plt.savefig(f"svg/eror_norma({ti}).svg", format='svg', bbox_inches='tight')
plt.show()

plt.plot(er_norm)
plt.grid(True)
plt.savefig(f"svg/eror_denorma({ti}).svg", format='svg', bbox_inches='tight')
plt.show()

print((sum(er) / len(er)))
print((sum(er) / len(er)) * (D_test.max - D_test.min))
