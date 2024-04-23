import matplotlib.pyplot as plt
import numpy as np


class Data:
    def __init__(self):
        self.data = []
        self.data_to_add = []
        self.data_set = []
        self.answer = []
        self.min = [0, 0, 0, 0, 0, 0]
        self.max = [0, 0, 0, 0, 0, 0]
        with open("water_df.csv") as f:
            while True:
                line = f.readline()
                if not line:
                    break

                temp = line.split(",")
                self.data.append([int(temp[0]), float(temp[1]),
                                  float(temp[2]), float(temp[3]),
                                  float(temp[4]), temp[5]])

        data_temp = [[], [], [], [], [], []]
        for i in self.data:
            for j in range(6):
                data_temp[j].append(i[j])

        for j in range(6):
            self.min[j] = min(data_temp[j])

        for j in range(6):
            self.max[j] = max(data_temp[j])

    def create_date_set(self, out_data, index_par, col, time, f):
        for i in range(col):
            temp = []
            for j in range(out_data):
                temp += (self.data[i + j][1:5])
            if f == 0:
                temp_valur = int(round(self.data[i + time + out_data][index_par] - self.data[i + out_data][index_par], 1) * 10)
                temp_list = [0 for col in range(200)]
                temp_list[temp_valur + 99] = 1
                self.answer.append(temp_list)
                self.data_to_add.append(self.data[i + out_data][index_par])
                self.data_set.append(temp)
            elif f == 1:
                temp_valur = (round(self.data[i + time + out_data][index_par] - self.data[i + out_data][index_par], 1))
                self.answer.append(temp_valur)
                self.data_to_add.append(self.data[i + out_data][index_par])
                self.data_set.append(temp)
            elif f == 2:
                temp_valur = (self.data[i + time + out_data][index_par] - self.data[i + out_data][index_par])
                self.answer.append(temp_valur)
                self.data_to_add.append(self.data[i + out_data][index_par])
                self.data_set.append(temp)

    def create_date_set_1_par(self, out_data, index_par, col, time):
        for i in range(col):
            temp = []
            for j in range(out_data):
                temp.append((self.data[i + j + 1][index_par] - self.min[index_par]) / (self.max[index_par] - self.min[index_par]))

            temp_valur = (self.data[i + time + out_data][index_par] - self.min[index_par]) / (self.max[index_par] - self.min[index_par])
            self.answer.append(temp_valur)
            self.data_set.append(temp)


def progress_bar(total, actual):
    l = "Progress: ["

    percent = round(actual * 100 / total, 2)
    painted = int(percent // 10)
    rest = int(10 - painted)

    l += "#" * painted
    l += "." * rest
    l += f"] - {percent}%"

    print('\r' + l, end='')


class Clean_data:
    len = 0
    max = 0
    min = 0

    def __init__(self, patch):
        self.extremes = []
        self.data_norm = []
        self.data = []
        self.data_set = []
        self.answer = []
        self.lying = []
        self.cheats_con = []
        self.cheats_start = []

        with open(f"{patch}") as f:
            while True:
                text_data = f.readline()
                if not text_data:
                    break
                self.data.append(float(text_data.replace(",", ".")))

        self.len = len(self.data)
        self.max = max(self.data)
        self.min = min(self.data)

    def normalization(self, out_data, time, prediction_length, start=0, length=None):   # эфективнее в 2 раза
        if length is None:
            len2 = self.len - out_data - time - prediction_length - start
        else:
            len2 = length

        raz = self.max - self.min

        self.data_norm = []

        for i in self.data:
            self.data_norm.append((i - self.min) / raz)

        for i in range(len2):
            self.answer.append(self.data_norm[(i + time + out_data + start):(i + time + out_data + prediction_length + start)])
            self.data_set.append(self.data_norm[(i + 1 + start):(i + 1 + out_data + start)])

            progress_bar(len2, i)

        print("\n")

    def derivative(self):
        der = []
        for i in range(len(self.data) - 1):
            der.append(self.data[i]-self.data[i + 1])
        return der

    def lying_test(self, l):
        for i in range(len(self.answer) - l):
            self.lying.append(self.answer[i + l])

    def smoothing(self, wind):
        data_temp = []
        for i in range(len(self.data) - wind):
            data_temp.append(sum(self.data[i : i + wind]) / wind)

        self.data = data_temp

        self.len = len(self.data)
        self.max = max(self.data)
        self.min = min(self.data)

    def normalization_data_set(self, out_data, time, prediction_length, start=0, length=None):
        if length is None:
            len2 = self.len - out_data - time - prediction_length - start
        else:
            len2 = length

        self.extremes = []

        for i in range(len2):
            max1 = max(self.data[(i + start):(i + out_data + time + prediction_length)]) + 0.00001
            min1 = min(self.data[(i + start):(i + out_data + time + prediction_length)])

            self.extremes.append([max1, min1])

            ds = []

            for j in range(out_data):
                ds.append((self.data[i + j + start] - min1) / (max1 - min1))

                # if ds[-1] > 1:
                #     print(ds[-1])

            self.data_set.append(ds)

            da = []

            for j in range(prediction_length):
                da.append((self.data[i + j + time + out_data + start] - min1) / (max1 - min1))

                # if da[-1] > 1:
                #     print(1)

            self.answer.append(da)


if __name__ == "__main__":
    a = Clean_data("D:\Pyton\prediction_on_the_water\data\orp\\1_hour.txt")
    a.normalization_data_set(10, 0, 10, 1, 1)

    print(a.extremes)

    print(a.data_set[0])

    plt.plot(a.data_set[0])
    plt.show()
