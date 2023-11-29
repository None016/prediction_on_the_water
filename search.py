import random
import keras
from keras.layers import Dense
import numpy as np
from data import Data


def search(l1, l2, l3, l4, bs, ep, vs, ids, loos, act, opt):

    model = keras.Sequential([
        Dense(l1, activation=act[0]),
        Dense(l2, activation=act[1]),
        Dense(l3, activation=act[2]),
        Dense(l4, activation=act[3]),
        Dense(1, activation=act[4]),
    ])

    model.compile(optimizer=opt,
                  loss=loos)

    D = Data()

    D.create_date_set_1_par(100 * 1, 4, 1_000, 30)

    m1 = 1_000 // 10

    x = np.array(D.data_set[0:m1 * 8])
    y = np.array(D.answer[0:m1 * 8])

    x_test = np.array(D.data_set[m1 * 8 + 1:1_000])
    y_test = np.array(D.answer[m1 * 8 + 1:1_000])

    print("Начало обучения")
    his = model.fit(x, y, batch_size=bs, epochs=ep, validation_split=vs)
    print("Конец обучения")
    # model.evaluate(x, y)

    nndata = model.predict(x_test)

    nndata_nor = []
    y_test_nor = []

    er = []

    for i in range(len(nndata)):
        nndata_nor.append(nndata[i] * (D.max[4] - D.min[4]) + D.min[4])
        y_test_nor.append(y_test[i] * (D.max[4] - D.min[4]) + D.min[4])
        er.append(abs(y_test[i] - nndata[i]))

    mae_test = np.sum(er) / len(er)
    print(mae_test)
    model.save(f"model2_orp(4)/model({mae_test})_{ids}")


min_ner = 1
max_ner = 2048
bs_min = 0
bs_max = 20
ep_min = 1
ep_max = 10
id = 0

loos_list = ["mae", "mape", "msle"]
len_loos_list = len(loos_list) - 1

act_list = ["linear", "relu"]
len_act_list = len(act_list) - 1

opt_list = ['Adam', "SGD", "RMSProp", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]
len_opt_list = len(opt_list) - 1

for i in range(1_000):
    l1 = random.randint(min_ner, max_ner)
    l2 = random.randint(min_ner, max_ner)
    l3 = random.randint(min_ner, max_ner)
    l4 = random.randint(min_ner, max_ner)

    bs = random.randint(bs_min, bs_max)

    ep = random.randint(ep_min, ep_max)

    vs = random.uniform(0, 0.8)
    loos = loos_list[random.randint(0, len_loos_list)]

    opt = opt_list[random.randint(0, len_opt_list)]

    act = []
    for i in range(5):
        act.append(act_list[random.randint(0, len_act_list)])

    id += 1

    with open(f"characteristics/characteristics.txt", "a") as f:
        f.write(f"id:{id};l1={l1};l2={l2};l3={l3};l4={l4};bs={2**bs};ep={ep};vs={vs};loos={loos};act={act};opt={opt}\n")

    search(l1, l2, l3, l4, 2**bs, ep, vs, id, loos, act, opt)
