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

'''
create_trainset_y1  这个是造训练数据集的（单任务）
create_trainset_y2  这个是造训练数据集的（多任务）
read_data  读训练数据集
read_label  读训练标签集
get_really_data and get_really_data_2 这个是生成数据排列组合的的list
get_really_list  这个是将特征数据放到list中，这样就可以放到神经网络里了呀
'''

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
    # print(t_1)
    # print(t_2)
    # print(t_3)
    # print(t_4)
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
    for i in range(len(data_test)):
        data_name_test.append(float(data_test[i][0]))
    f_test.close()
    return data_name_test

def get_really_data(all_data,number):
    #这个函数会中间调用get_really_data_2
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
    #这个是将排列组合的list出现，return给get_really_data函数。
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

def get_really_list(all_data_2,model_number):
    # 这个是将特征数据放到list中，这样就可以放到神经网络里了呀
    x_list = []
    for i in range(model_number):
        x_list.append(np.asarray(all_data_2[i][0]))
    return x_list

if __name__ == "__main__":
    # create_trainset_y1() #这个是造训练数据的
    all_data = read_data()
    all_label = read_label()
    train_y = all_label
    train_y = np.asarray(train_y)
    all_data = np.asarray(all_data)
    all_data_2 = get_really_data(all_data,all_data.shape[1])

    model_number = len(all_data_2) #这个是个算一下有多少个feature的排列组合。
    model_number_list = []
    for i in range(model_number):
        model_number_list.append(len(all_data_2[i][0][0])) #这个是将数据集的维数标出来。

    '''
    这个是搞模型的
    特征模型是两层全连接。
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
    # model = tf.nn.atrous_conv2d()
    feature_model = [None] * model_number
    for i in range(model_number):
        feature_model[i] = Model(input=feature[i],output=Dense3[i])
    feature_model[0].summary
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])
    plot_model(model, to_file="model.png", show_shapes=True)
    #这里可以写一个函数，让其输出list。
    x_list = get_really_list(all_data_2,model_number)

    model.fit(x=x_list, y=[train_y], epochs=300,batch_size=50)

    #这个是进行查看最后一层的选择的权重。
    weight_Dense_1 , bias_Dense_1 = model.get_layer('output').get_weights()
    print(weight_Dense_1)

    #这个现在模型以及弄好了，我们现在需要进行画图了，这个画图也就是用均匀点画图。
    x1_test = []
    for i in range(100):
        x1_test.append(0.01 * i)

    x1_test = np.asarray(x1_test)
    y1_test = feature_model[0].predict(x1_test)
    # print(x1_test)
    # print(y1_test)
    y1_plt = []
    for i in range(len(y1_test)):
        y1_plt.append(y1_test[i])
    y1_mean = np.mean(y1_plt)
    for i in range(len(y1_plt)):
        y1_plt[i] = y1_plt[i] - y1_mean
    plt.scatter(x1_test,y1_plt)
    # plt.show()
    plt.savefig('x1.jpg')
    plt.close()

    x2_test = []
    for i in range(200):
        x2_test.append(-1 + 0.01 * i)
    x2_test = np.asarray(x2_test)
    y2_test = feature_model[1].predict(x2_test)
    y2_plt = []
    for i in range(len(y2_test)):
        y2_plt.append(y2_test[i])
    y2_mean = np.mean(y2_plt)
    for i in range(len(y2_plt)):
        y2_plt[i] = y2_plt[i] - y2_mean
    plt.scatter(x2_test, y2_plt)
    # plt.show()
    plt.savefig('x2.jpg')
    plt.close()

    x3_test = []
    for i in range(200):
        x3_test.append(-1 + 0.01 * i)
    x3_test = np.asarray(x3_test)
    y3_test = feature_model[2].predict(x3_test)
    y3_plt = []
    for i in range(len(y3_test)):
        y3_plt.append(y3_test[i])
    y3_mean = np.mean(y3_plt)
    for i in range(len(y3_plt)):
        y3_plt[i] = y3_plt[i] - y3_mean
    plt.scatter(x3_test, y3_plt)
    # plt.show()
    plt.savefig('x3.jpg')
    plt.close()
