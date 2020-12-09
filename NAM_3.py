import random
from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.merge import concatenate
import keras
import numpy as np
from itertools import combinations
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.utils import plot_model
# import tensorflow.compat.v1 as tf #使用1.0版本的方法
# tf.disable_v2_behavior() #禁用2.0版本的方法


def create_trainset_y1():
    #这个是造单任务的数据
    x1 = []
    x2 = []
    x3 = []
    y = []
    t = 0
    for i in range(1400):
        t1 = random.uniform(0, 1)
        x1.append(t1)
        t2 = random.uniform(-1, 1)
        x2.append(t2)
        t3 = random.uniform(-1, 1)
        x3.append(t3)
        if t1 > 0.5 and t2 * t3 > 0:
            y.append(1)
            t += 1
        else:
            y.append(0)
    print(t)
    with open('nam_test_train_x1.txt', 'w') as f:
        for i in range(1400):
            test_str = str(x1[i]) + " " + str(x2[i]) + " " + str(x3[i]) + "\n"
            f.write(test_str)
    with open('nam_test_train_y1.txt', 'w') as f:
        for i in range(1400):
            test_str = str(y[i]) + "\n"
            f.write(test_str)
def create_trainset_y11():
    #这个是造单任务的数据
    x1 = []
    x2 = []
    x3 = []
    y = []
    t = 0
    for i in range(400):
        t1 = random.uniform(0, 1)
        x1.append(t1)
        t2 = random.uniform(-1, 1)
        x2.append(t2)
        t3 = random.uniform(-1, 1)
        x3.append(t3)
        if t1 > 0.5 and t2 * t3 > 0:
            y.append(1)
            t += 1
        else:
            y.append(0)
    print(t)
    with open('nam_test_train_x11.txt', 'w') as f:
        for i in range(1400):
            test_str = str(x1[i]) + " " + str(x2[i]) + " " + str(x3[i]) + "\n"
            f.write(test_str)
    with open('nam_test_train_y11.txt', 'w') as f:
        for i in range(1400):
            test_str = str(y[i]) + "\n"
            f.write(test_str)

def create_trainset_y2():
    #这个是造多任务的数据
    x1 = []
    x2 = []
    x3 = []
    y = []
    t_1 = 0
    t_2 = 0
    t_3 = 0
    t_4 = 0
    for i in range(1400):
        t1 = random.uniform(0, 1)
        x1.append(t1)
        t2 = random.uniform(-1, 1)
        x2.append(t2)
        t3 = random.uniform(-1, 1)
        x3.append(t3)
        if t1 > 0.5 and t2 * t3 > 0:
            y.append(1)
            t_1 += 1
        elif t1<0.5 and t2 * t3 >0:
            y.append(2)
            t_2 += 1
        elif t1>0.5 and t2*t3<0:
            y.append(3)
            t_3 += 1
        else:
            y.append(4)
            t_4 += 1
    print(t_1)
    print(t_2)
    print(t_3)
    print(t_4)
    with open('nam_test_train_y2.txt', 'w') as f:
        for i in range(1400):
            test_str = str(x1[i]) + " " + str(x2[i]) + " " + str(x3[i]) + " " + str(y[i]) + "\n"
            f.write(test_str)

def read_data():
    f_test = open("nam_test_train_x1.txt",'r')
    data_test = f_test.readlines()
    data_name_test = []
    for i in range(len(data_test)):
        data_name_test.append([])
    for i in range(len(data_test)):
        data_test[i] = data_test[i].split(' ')
        data_test[i][-1] = data_test[i][-1][:-2] #这里需要注意一下
        for j in range(len(data_test[i])):
            if data_test[i][j] != '':
                data_name_test[i].append(float(data_test[i][j]))
    f_test.close()
    return data_name_test

def read_label():
    f_test = open("nam_test_train_y1.txt", 'r')
    data_test = f_test.readlines()
    data_name_test = []
    # for i in range(len(data_test)):
    #     data_name_test.append([])
    for i in range(len(data_test)):
        # data_test[i][0] = data_test[i][0][:-2]
        data_name_test.append(float(data_test[i][0]))
        # data_test[i] = data_test[i].split(' ')
        # data_test[i][-1] = data_test[i][-1][:-2]  # 这里需要注意一下
        # for j in range(len(data_test[i])):
        #     if data_test[i][j] != '':
        #         data_name_test[i].append(data_test[i][j])
    f_test.close()
    return data_name_test

def get_really_data(all_data,number):
    t_n = []
    for i in range(number):
        t_n.append(i)
    all_data_3 = []
    sum = 0
    for i in range(number):
        for j in combinations(t_n,i+1):
            all_data_3.append([])
            # print(j) # j=0,1,2,3,(0,1),(0,2)...
            for k in range(len(j)):
                all_data_3[sum].append(get_really_data_2(all_data,j))
            sum += 1
    return all_data_3
def get_really_data_2(all_data,number_list):
    all_data_4 = []
    for i in range(len(all_data)):
        all_data_4.append([])
        for j in range(len(number_list)):
            all_data_4[i].append([])
    for i in range(len(all_data)):
        for j in range(len(number_list)):
            all_data_4[i][j] = all_data[i][number_list[j]]
    # print(all_data_4)
    return all_data_4

if __name__ == "__main__":
    # create_trainset_y1() #这个是造数据的

    all_data = read_data()
    all_label = read_label()
    train_y = all_label
    train_y = np.asarray(train_y)
    all_data = np.asarray(all_data)
    all_data_2 = get_really_data(all_data,all_data.shape[1])

    create_trainset_y11() #这个是造数据的
    all_data1 = read_data()
    all_label1 = read_label()
    train_y1 = all_label1
    train_y1 = np.asarray(train_y1)
    all_data1 = np.asarray(all_data1)
    all_data_21 = get_really_data(all_data1, all_data1.shape[1])


    '''
    现在缺一个可以将list的一列放到函数里面的方法，。。。。再问问其他人把。
    '''

    model_number = len(all_data_2)
    model_number_list = []
    for i in range(model_number):
        model_number_list.append(len(all_data_2[i][0][0]))
    # print(model_number_list)
    '''
    这个是搞模型的
    :return:
    '''
    feature = [None] * model_number
    Dense1 = [None] * model_number
    Dense2 = [None] * model_number
    Dense3 = [None] * model_number
    for i in range(model_number):
        feature[i] = Input(shape=(model_number_list[i],),name="feature_" + str(i))
        Dense1[i] = Dense(1024, activation='relu', name="Dense" + str(i) + "_1")(feature[i])
        Dense2[i] = Dense(1024, activation='relu', name="Dense" + str(i) + "_2")(Dense1[i])
        Dense3[i] = Dense(1, activation='sigmoid', name="Dense" + str(i) + "_3")(Dense2[i])
    merge = concatenate([Dense3[i] for i in range(model_number)])
    output = Dense(1,activation='sigmoid',name="output")(merge)
    model = Model(input=[feature[i] for i in range(model_number)], output=output)

    feature_model = [None] * model_number
    for i in range(model_number):
        feature_model[i] = Model(input=feature[i],output=Dense3[i])

    model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])

    train_x1 = np.asarray(all_data_2[0][0])
    train_x2 = np.asarray(all_data_2[1][0])
    train_x3 = np.asarray(all_data_2[2][0])
    train_x12 = np.asarray(all_data_2[3][0])
    train_x13 = np.asarray(all_data_2[4][0])
    train_x23 = np.asarray(all_data_2[5][0])
    train_x123 = np.asarray(all_data_2[6][0])

    model.fit(x=[train_x1, train_x2, train_x3, train_x12, train_x13, train_x23, train_x123], y=[train_y], epochs=300,batch_size=50)
    # model.fit(x=[np.asarray(all_data_2[i][0]) for i in range(6)], y=[train_y], epochs=300,
    #           batch_size=50)

    test_x1 = np.asarray(all_data_21[0][0])
    test_x2 = np.asarray(all_data_21[1][0])
    test_x3 = np.asarray(all_data_21[2][0])
    test_x12 = np.asarray(all_data_21[3][0])
    test_x13 = np.asarray(all_data_21[4][0])
    test_x23 = np.asarray(all_data_21[5][0])
    test_x123 = np.asarray(all_data_21[6][0])

    test_y = model.predict(x=[test_x1, test_x2, test_x3, test_x12, test_x13, test_x23, test_x123])

    # x1_test = []
    # for i in range(100):
    #     x1_test.append(0.01 * i)
    #
    # x1_test = np.asarray(x1_test)
    # y1_test = feature_model[0].predict(x1_test)
    # # print(x1_test)
    # # print(y1_test)
    # y1_plt = []
    # for i in range(len(y1_test)):
    #     y1_plt.append(y1_test[i])
    # y1_mean = np.mean(y1_plt)
    # for i in range(len(y1_plt)):
    #     y1_plt[i] = y1_plt[i] - y1_mean
    # plt.scatter(x1_test,y1_plt)
    # plt.show()
    # train_all_set = all_data[:800]
    # val_all_set = all_data[800:1000]
    # test_all_set = all_data[1000:]
    #
    # train_x1 = []
    # train_x2 = []
    # train_x3 = []
    # train_x12 = []
    # train_x13 = []
    # train_x23 = []
    # train_x123 = []
    # train_y = []
    # for i in range(len(train_all_set)):
    #     train_x1.append(float(train_all_set[i][0]))
    #     train_x2.append(float(train_all_set[i][1]))
    #     train_x3.append(float(train_all_set[i][2]))
    #     train_x12.append((float(train_all_set[i][0]),float(train_all_set[i][1])))
    #     train_x13.append((float(train_all_set[i][0]), float(train_all_set[i][2])))
    #     train_x23.append((float(train_all_set[i][1]), float(train_all_set[i][2])))
    #     train_x123.append((float(train_all_set[i][0]), float(train_all_set[i][1]),float(train_all_set[i][2])))
    #     train_y.append(float(train_all_set[i][3]))
    # train_x1 = np.asarray(train_x1)
    # train_x2 = np.asarray(train_x2)
    # train_x3 = np.asarray(train_x3)
    # train_x12 = np.asarray(train_x12)
    # train_x13 = np.asarray(train_x13)
    # train_x23 = np.asarray(train_x23)
    # train_x123 = np.asarray(train_x123)
    # train_y = np.asarray(train_y)
    #
    # val_x1 = []
    # val_x2 = []
    # val_x3 = []
    # val_x12 = []
    # val_x13 = []
    # val_x23 = []
    # val_x123 = []
    # val_y = []
    # for i in range(len(val_all_set)):
    #     val_x1.append(float(val_all_set[i][0]))
    #     val_x2.append(float(val_all_set[i][1]))
    #     val_x3.append(float(val_all_set[i][2]))
    #     val_x12.append((float(val_all_set[i][0]),float(val_all_set[i][1])))
    #     val_x13.append((float(val_all_set[i][0]), float(val_all_set[i][2])))
    #     val_x23.append((float(val_all_set[i][1]), float(val_all_set[i][2])))
    #     val_x123.append((float(val_all_set[i][0]),float(val_all_set[i][1]),float(val_all_set[i][2])))
    #     val_y.append(float(val_all_set[i][3]))
    # val_x1 = np.asarray(val_x1)
    # val_x2 = np.asarray(val_x2)
    # val_x3 = np.asarray(val_x3)
    # val_x12 = np.asarray(val_x12)
    # val_x13 = np.asarray(val_x13)
    # val_x23 = np.asarray(val_x23)
    # val_x123 = np.asarray(val_x123)
    # val_y = np.asarray(val_y)
    #
    # test_x1 = []
    # test_x2 = []
    # test_x3 = []
    # test_x12 = []
    # test_x13 = []
    # test_x23 = []
    # test_x123 = []
    # test_y = []
    # for i in range(len(test_all_set)):
    #     test_x1.append(float(test_all_set[i][0]))
    #     test_x2.append(float(test_all_set[i][1]))
    #     test_x3.append(float(test_all_set[i][2]))
    #     test_x12.append((float(test_all_set[i][0]),float(test_all_set[i][1])))
    #     test_x13.append((float(test_all_set[i][0]), float(test_all_set[i][2])))
    #     test_x23.append((float(test_all_set[i][1]), float(test_all_set[i][2])))
    #     test_x123.append((float(test_all_set[i][0]),float(test_all_set[i][1]),float(test_all_set[i][2])))
    #     test_y.append(float(test_all_set[i][3]))
    # test_x1 = np.asarray(test_x1)
    # test_x2 = np.asarray(test_x2)
    # test_x3 = np.asarray(test_x3)
    # test_x12 = np.asarray(test_x12)
    # test_x13 = np.asarray(test_x13)
    # test_x23 = np.asarray(test_x23)
    # test_x123 = np.asarray(test_x123)
    # test_y = np.asarray(test_y)
    #
    # # 上面我们搞定了模拟数据，下面就是搭建模型了！！
    #
    #
    # feature_1 = Input(shape=(1,),name="feature_1")
    # Dense11 = Dense(1024,activation='relu',name="Dense11")(feature_1)
    # Dense12 = Dense(1024, activation='relu', name="Dense12")(Dense11)
    # Dense13 = Dense(1,activation='sigmoid',name="Dense13")(Dense12)
    #
    # feature_2 = Input(shape=(1,),name="feature_2")
    # Dense21 = Dense(1024,activation='relu',name="Dense21")(feature_2)
    # Dense22 = Dense(1024, activation='relu', name="Dense22")(Dense21)
    # Dense23 = Dense(1,activation='sigmoid',name="Dense23")(Dense22)
    #
    # feature_3 = Input(shape=(1,),name="feature_3")
    # Dense31 = Dense(1024,activation='relu',name="Dense31")(feature_3)
    # Dense32 = Dense(1024, activation='relu', name="Dense32")(Dense31)
    # Dense33 = Dense(1,activation='sigmoid',name="Dense33")(Dense32)
    #
    # feature_4 = Input(shape=(2,), name="feature_4")
    # Dense41 = Dense(1024, activation='relu', name="Dense41")(feature_4)
    # Dense42 = Dense(1024, activation='relu', name="Dense42")(Dense41)
    # Dense43 = Dense(1, activation='sigmoid', name="Dense43")(Dense42)
    #
    # feature_5 = Input(shape=(2,),name="feature_5")
    # Dense51 = Dense(1024,activation='relu',name="Dense51")(feature_5)
    # Dense52 = Dense(1024, activation='relu', name="Dense52")(Dense51)
    # Dense53 = Dense(1,activation='sigmoid',name="Dense53")(Dense52)
    #
    # feature_6 = Input(shape=(2,),name="feature_6")
    # Dense61 = Dense(1024,activation='relu',name="Dense61")(feature_6)
    # Dense62 = Dense(1024, activation='relu', name="Dense62")(Dense61)
    # Dense63 = Dense(1,activation='sigmoid',name="Dense63")(Dense62)
    #
    # feature_7 = Input(shape=(3,),name="feature_7")
    # Dense71 = Dense(1024,activation='relu',name="Dense71")(feature_7)
    # Dense72 = Dense(1024, activation='relu', name="Dense72")(Dense71)
    # Dense73 = Dense(1,activation='sigmoid',name="Dense73")(Dense72)
    #
    # merge = concatenate([Dense13,Dense23,Dense33,Dense43,Dense53,Dense63,Dense73])
    # output = Dense(1,activation='sigmoid',name="output")(merge)
    #
    # model = Model(input=[feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7], output=output)
    #
    # feature_model_1 = Model(input=feature_1,output=Dense13)
    # feature_model_2 = Model(input=feature_2, output=Dense23)
    # feature_model_3 = Model(input=feature_3, output=Dense33)
    # feature_model_4 = Model(input=feature_4,output=Dense43)
    # feature_model_5 = Model(input=feature_5, output=Dense53)
    # feature_model_6 = Model(input=feature_6, output=Dense63)
    # feature_model_7 = Model(input=feature_7, output=Dense73)
    #
    # model.summary()
    # # print("model finish!")
    #
    # opt = keras.optimizers.Adam(learning_rate=0.01)
    #
    # model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])
    # # plot_model(model, to_file="model.png", show_shapes=True)
    #
    #
    #
    # model.fit(x=[train_x1,train_x2,train_x3,train_x12,train_x13,train_x23,train_x123],y=[train_y],epochs=300,batch_size=50)
    # # model.fit([headline_data, additional_data], [labels, labels], epochs=50, batch_size=32)
    #
    #
    # x1_test = []
    # for i in range(200):
    #     x1_test.append(0.01 * i)
    #
    # x1_test = np.asarray(x1_test)
    # y1_test = feature_model_1.predict(x1_test)
    # # print(x1_test)
    # # print(y1_test)
    # y1_plt = []
    # for i in range(len(y1_test)):
    #     y1_plt.append(y1_test[i])
    # y1_mean = np.mean(y1_plt)
    # for i in range(len(y1_plt)):
    #     y1_plt[i] = y1_plt[i] - y1_mean
    # plt.scatter(x1_test,y1_plt)
    # # plt.show()
    # # plt.savefig('x1.jpg')
    # plt.close()
    #
    # x2_test = []
    # for i in range(200):
    #     x2_test.append(-1 + 0.01 * i)
    # x2_test = np.asarray(x2_test)
    # y2_test = feature_model_2.predict(x2_test)
    # y2_plt = []
    # for i in range(len(y2_test)):
    #     y2_plt.append(y2_test[i])
    # y2_mean = np.mean(y2_plt)
    # for i in range(len(y2_plt)):
    #     y2_plt[i] = y2_plt[i] - y2_mean
    # plt.scatter(x2_test, y2_plt)
    # # plt.show()
    # # plt.savefig('x2.jpg')
    # plt.close()
    #
    # x3_test = []
    # for i in range(200):
    #     x3_test.append(-1 + 0.01 * i)
    # x3_test = np.asarray(x3_test)
    # y3_test = feature_model_3.predict(x3_test)
    # y3_plt = []
    # for i in range(len(y3_test)):
    #     y3_plt.append(y3_test[i])
    # y3_mean = np.mean(y3_plt)
    # for i in range(len(y3_plt)):
    #     y3_plt[i] = y3_plt[i] - y3_mean
    # plt.scatter(x3_test, y3_plt)
    # # plt.show()
    # # plt.savefig('x3.jpg')
    # plt.close()
    #
    # x4_test = []
    # for i in range(200):
    #     for j in range(200):
    #         x4_test.append((0.01 * i, -1 + 0.01 * j))
    # x4_1_plt = []
    # x4_2_plt = []
    # for i in range(len(x4_test)):
    #     x4_1_plt.append(x4_test[i][0])
    #     x4_2_plt.append(x4_test[i][1])
    # x4_test = np.asarray(x4_test)
    # y4_test = feature_model_4.predict(x4_test)
    # y4_plt = []
    # for i in range(len(y4_test)):
    #     y4_plt.append(y4_test[i])
    # y4_mean = np.mean(y4_plt)
    # for i in range(len(y4_plt)):
    #     y4_plt[i] = y4_plt[i] - y4_mean
    # fig = plt.figure()
    # ax1 = Axes3D(fig)
    # ax1.scatter(x4_1_plt,x4_2_plt,y4_plt)
    # ax1.set_xlabel('x1 Label')
    # ax1.set_ylabel('x2 Label')
    # ax1.set_zlabel('y Label')
    # plt.show()
    #
    # x5_test = []
    # for i in range(200):
    #     for j in range(200):
    #         x5_test.append((0.01 * i, -1 + 0.01 * j))
    # x5_1_plt = []
    # x5_2_plt = []
    # for i in range(len(x5_test)):
    #     x5_1_plt.append(x5_test[i][0])
    #     x5_2_plt.append(x5_test[i][1])
    # x5_test = np.asarray(x5_test)
    # y5_test = feature_model_5.predict(x5_test)
    # y5_plt = []
    # for i in range(len(y5_test)):
    #     y5_plt.append(y5_test[i])
    # y5_mean = np.mean(y5_plt)
    # for i in range(len(y5_plt)):
    #     y5_plt[i] = y5_plt[i] - y5_mean
    # fig = plt.figure()
    # ax1 = Axes3D(fig)
    # ax1.scatter(x5_1_plt,x5_2_plt,y5_plt)
    # ax1.set_xlabel('x2 Label')
    # ax1.set_ylabel('x3 Label')
    # ax1.set_zlabel('y Label')
    # plt.show()
    #
    # x6_test = []
    # for i in range(200):
    #     for j in range(200):
    #         x6_test.append((-1 + 0.01 * i, -1 + 0.01 * j))
    # x6_1_plt = []
    # x6_2_plt = []
    # for i in range(len(x6_test)):
    #     x6_1_plt.append(x6_test[i][0])
    #     x6_2_plt.append(x6_test[i][1])
    # x6_test = np.asarray(x6_test)
    # y6_test = feature_model_6.predict(x6_test)
    # y6_plt = []
    # for i in range(len(y6_test)):
    #     y6_plt.append(y6_test[i])
    # y6_mean = np.mean(y6_plt)
    # for i in range(len(y6_plt)):
    #     y6_plt[i] = y6_plt[i] - y6_mean
    # fig = plt.figure()
    # ax1 = Axes3D(fig)
    # ax1.scatter(x6_1_plt,x6_2_plt,y6_plt)
    # ax1.set_xlabel('x2 Label')
    # ax1.set_ylabel('x3 Label')
    # ax1.set_zlabel('y Label')
    # plt.show()
    #
    # # x7_test = []
    # # for i in range(200):
    # #     x7_test.append((0.01 * i, -1 + 0.01 * i,-1 + 0.01 * i))
    # # x7_1_plt = []
    # # x7_2_plt = []
    # # x7_3_plt = []
    # # for i in range(len(x7_test)):
    # #     x7_1_plt.append(x7_test[i][0])
    # #     x7_2_plt.append(x7_test[i][1])
    # #     x7_3_plt.append(x7_test[i][2])
    # # x7_test = np.asarray(x7_test)
    # # y7_test = feature_model_7.predict(x7_test)
    # # y7_plt = []
    # # for i in range(len(y7_test)):
    # #     y7_plt.append(y7_test[i])
    # # y7_mean = np.mean(y7_plt)
    # # for i in range(len(y7_plt)):
    # #     y7_plt[i] = y7_plt[i] - y7_mean
    # # fig = plt.figure()
    # # ax1 = Axes3D(fig)
    # # ax1.scatter(x7_1_plt,x7_2_plt,y7_plt)
    # # plt.show()
    #
    #
