import keras
from keras.layers import *
import numpy as np
import matplotlib.pyplot as plt

from data import *


model = keras.Sequential([
    LSTM(600, input_shape=(1260, 1), return_sequences=True),
    LSTM(64),
    # Conv1D(16, 16, padding='same', activation='relu', input_shape=(2880, 1)),
    # Conv1D(32, 16, padding='same', activation='relu'),
    # MaxPooling1D(4),
    # Conv1D(64, 16, padding='same', activation='relu'),
    # MaxPooling1D(8),
    # Flatten(),
    Dense(700, activation="sigmoid"),
    Dense(256, activation="relu"),
    Dense(256, activation="relu"),
    Dense(256, activation="relu"),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(540, activation="linear"),
])

model.compile(optimizer='adam',
              loss='mse',    # mean_squared_error
              metrics=["mean_squared_error", "mae", "mean_absolute_percentage_error", "mse"])
# __________________^^^^ "mean_squared_error", "msle" <<< плохо работает стоит попробовать без нормализации
# metrics=['accuracy']

print("_____________________________")

D = Clean_data("D:\Pyton\prediction_on_the_water\data\weekly_data\o2.txt")
D_test = Clean_data("data/o2/3hour.txt")

print("_____________________________")

# D.smoothing(1000)
D.normalization(1260, 1, 540, 0, 150_000)

print("_____________________________")

D_test.smoothing(1000)
D_test.normalization(1260, 1, 540)

print(f"max {D.max}")
print(f"min {D.min}")

print(f"max {D_test.max}")
print(f"min {D_test.min}")

x = np.array(D.data_set)
y = np.array(D.answer)

x_test = np.array(D_test.data_set)
y_test = np.array(D_test.answer)

l = 0

print("Начало обучения")
his = model.fit(x, y, batch_size=64, epochs=1, validation_split=0.2)
print("Конец обучения")


model.save("model/o2_30min")

nndata = model.predict(x_test)

plt.plot(y_test[0])
plt.plot(nndata[0])
plt.grid(True)
plt.show()

nndata_nor = []
y_test_nor = []

er = []

for i in range(len(nndata)):
    nndata_nor.append(nndata[i] * (D_test.max - D_test.min) + D_test.min)
    y_test_nor.append(y_test[i] * (D_test.max - D_test.min) + D_test.min)
    er.append(abs(y_test[i] - nndata[i]))

plt.plot(np.array(y_test_nor))
plt.plot(np.array(nndata_nor))
plt.grid(True)
plt.show()

plt.plot(er)
plt.grid(True)
plt.show()

mae_test = np.sum(er) / len(er)
print(mae_test)
print(mae_test * (D_test.max - D_test.min))
