import matplotlib.pyplot as plt


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
                self.data.append([int(temp[0]), float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4]), temp[5]])

        data_temp = [[], [], [], [], [], []]
        for i in self.data:
            data_temp[0].append(i[0])
            data_temp[1].append(i[1])
            data_temp[2].append(i[2])
            data_temp[3].append(i[3])
            data_temp[4].append(i[4])
            data_temp[5].append(i[5])

        self.min[0] = min(data_temp[0])
        self.min[1] = min(data_temp[1])
        self.min[2] = min(data_temp[2])
        self.min[3] = min(data_temp[3])
        self.min[4] = min(data_temp[4])
        self.min[5] = min(data_temp[5])

        self.min[0] = max(data_temp[0])
        self.min[1] = max(data_temp[1])
        self.min[2] = max(data_temp[2])
        self.min[3] = max(data_temp[3])
        self.min[4] = max(data_temp[4])
        self.min[5] = max(data_temp[5])

    # количество секунд данных, индекс ответного значения, количество данных
    # через какое количество времени мы предсказываем данные
    def create_date_set(self, out_data, index_par, col, time, f):
        for i in range(col):
            temp = []
            for j in range(out_data):
                temp += (self.data[i + j][1:5])
            if f == 0:
                # temp_answer = []
                # temp_answer.append([self.data[i + time + col][index_par] - self.data[i + col][index_par]])
                # _____________________________________________________ - self.data[i + out_data][index_par]
                temp_valur = int(round(self.data[i + time + out_data][index_par] - self.data[i + out_data][index_par], 1) * 10)
                # print(temp_valur)
                temp_list = [0 for col in range(200)]
                temp_list[temp_valur + 99] = 1
                # print(len(temp_list), " - ", temp_list[temp_valur + 50])
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

    def f(self, out_data, col, time):
        for i in range(col):
            temp = []
            for j in range(out_data):
                temp.append(float((i + j) ** 2))
            self.data_set.append(temp)
            self.answer.append([(i + out_data + col) ** 2])


if __name__ == "__main__":
    a = Data()
    a.create_date_set_1_par(4000, 3, 1, 1)
    b = Data()
    b.create_date_set_1_par(4000, 2, 1, 1)
    # a.f(10, 100, 5)

    print(a.answer)
    print(b.answer)
    # print(a.data_set[0])

    # plt.plot(a.data_set[0])
    plt.plot(b.data_set[0])
    plt.grid(True)
    plt.show()
