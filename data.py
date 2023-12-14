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

        self.min[3] = 50
        self.max[3] = 600

        # self.min[3] = 163.5
        # self.max[3] = 351.6
        # self.min = [163.5, 163.5, 163.5, 163.5, 163.5, 163.5]
        # self.max = [351.6, 351.6, 351.6, 351.6, 351.6, 351.6]

        # for ORP

    # количество секунд данных, индекс ответного значения, количество данных
    # через какое количество времени мы предсказываем данные
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


class Clean_data:
    len = 0
    max = 0
    min = 0

    def __init__(self, patch):
        self.data = []
        self.data_set = []
        self.answer = []
        self.lying = []

        with open(f"{patch}") as f:
            while True:
                text_data = f.readline()
                if not text_data:
                    break
                self.data.append(float(text_data.replace(",", ".")))

        self.len = len(self.data)
        self.max = max(self.data)
        self.min = min(self.data)
        self.min = 50
        self.max = 600

    def normalization(self, out_data, time):
        for i in range(self.len - out_data - time):
            temp = []
            for j in range(out_data):
                temp.append((self.data[i + j + 1] - self.min) / (
                            self.max - self.min))

            temp_valur = (self.data[i + time + out_data] - self.min) / (
                        self.max - self.min)
            self.answer.append(temp_valur)
            self.data_set.append(temp)

    def lying_test(self, l):
        for i in range(len(self.answer) - l):
            self.lying.append(self.answer[i + l])


if __name__ == "__main__":
    a = Clean_data("data\orp\\3_day.txt")
    a.normalization(1, 1000)
    a.lying_test(1000)

    plt.plot(a.answer)
    plt.plot(a.lying)
    plt.grid(True)
    plt.show()

