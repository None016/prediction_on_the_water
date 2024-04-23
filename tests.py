from data import *
import numpy as np


def MAbsolutePercentageError(y_true, y_predict):
    return sum(100 * abs((y_true - y_predict) / y_true)) / len(y_true)


if __name__ == "__main__":
    print(MAbsolutePercentageError(np.array([100, 100]), np.array([50, 50])))
    # test_several_exits("model/o2_3hour", "data\o2\\3day.txt", 1000, 100, 1, 100)

