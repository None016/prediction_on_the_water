import keras
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

from data import *


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

inpex_par = 3
col = 400_000


D_test = Clean_data("data\do\\3day.txt")

D_test.normalization(100, 100)

x = np.array(D_test.data_set)
y = np.array(D_test.answer)


print("Начало обучения")
his = model.fit(x, y, batch_size=32, epochs=4, validation_split=0.2)
print("Конец обучения")


model.save("model/DO_")
