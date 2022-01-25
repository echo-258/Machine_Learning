import struct
from numpy import *
import numpy as np
import time
from tqdm import tqdm


def read_image(file_name):
    # 先用二进制方式把文件都读进来
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()
    offset = 0
    head = struct.unpack_from('>IIII', file_content, offset)  # 取前4个整数，返回一个元组
    offset += struct.calcsize('>IIII')
    imgNum = head[1]  # 图片数
    rows = head[2]  # 宽度
    cols = head[3]  # 高度

    images = np.empty((imgNum, 784))  # 数组内所有元素先置为空
    image_size = rows * cols  # 单个图片的大小
    fmt = '>' + str(image_size) + 'B'  # 单个图片的format

    for i in tqdm(range(imgNum)):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset))
        # images[i] = np.array(struct.unpack_from(fmt, file_content, offset)).reshape((rows, cols))
        offset += struct.calcsize(fmt)
    return images


# 读取标签
def read_label(file_name):
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中

    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')

    labelNum = head[1]  # label数
    # print(labelNum)
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)


def normalize(data):
    m = data.shape[0]
    n = np.array(data).shape[1]
    for i in range(m):
        for j in range(n):
            if data[i, j] != 0:
                data[i, j] = 1
            else:
                data[i, j] = 0
    return data


def loadDataSet():
    # mnist
    train_x_filename = "train-images-idx3-ubyte"
    train_y_filename = "train-labels-idx1-ubyte"
    test_x_filename = "t10k-images-idx3-ubyte"
    test_y_filename = "t10k-labels-idx1-ubyte"


    train_x = read_image(train_x_filename)  # 训练集
    train_y = read_label(train_y_filename)  # 训练集label
    test_x = read_image(test_x_filename)  # 测试集
    test_y = read_label(test_y_filename)  # 测试集label

    train_x = normalize(train_x)
    test_x = normalize(test_x)

    train_x=train_x[0:1000,:]
    train_y=train_y[0:1000]
    test_x=test_x[0:100,:]
    test_y=test_y[0:100]

    return train_x, test_x, train_y, test_y


# 计算先验概率及条件概率
def train_model(train_x, train_y, classNum):  # classNum：有10个类别
    m = train_x.shape[0]
    n = train_x.shape[1]

    prior_probability = np.zeros(classNum)  # 先验概率
    label_cnt = np.zeros(classNum)
    conditional_probability = np.zeros((classNum, n, 2))  # 条件概率
    conditional_cnt = np.zeros((classNum, n, 2))  # 条件概率
    # 计算先验概率和条件概率
    for i in range(m):  # m是图片数量
        img = train_x[i]  # img是第i个图片，是1*n的行向量
        label = train_y[i]  # label是第i个图片对应的label
        label_cnt[label] += 1  # 统计每类label的数量，之后除以图片数量就是某个类的先验概率
        for j in range(n):  # n是特征数，共784个
            temp = int(img[j])  # img[j]是0.0，需要先转为整数
            conditional_cnt[label][j][temp] += 1
            # 统计的是类为label的，在每个列中为1或者0的行数为多少，img[j]的值要么就是0要么就是1，计算条件概率

    for i in range(classNum):
        prior_probability[i] = label_cnt[i] / m

    # 计算概率
    for i in range(classNum):
        for j in range(n):
            # 计算0，1像素点对应的条件概率，
            conditional_probability[i][j][0] = conditional_cnt[i][j][0] / label_cnt[i]
            conditional_probability[i][j][1] = conditional_cnt[i][j][1] / label_cnt[i]
    return prior_probability, conditional_probability


# 对给定的x，计算先验概率和条件概率的乘积
def cal_probability(img, label, prior_probability, conditional_probability):
    probability = prior_probability[label]  # 先验概率
    n = img.shape[0] # 特征数
    # print(n)
    for i in range(n):
        probability *= conditional_probability[label][i][int(img[i])]

    return probability


# 确定实例x的类
def predict(test_x, test_y, prior_probability, conditional_probability):
    predict_y = []
    m = test_x.shape[0]
    n = test_x.shape[1]
    for i in range(m):
        img = np.array(test_x[i])  # img已经是二值化以后的列向量
        label = test_y[i]
        max_label = 0
        max_probability = cal_probability(img, 0, prior_probability, conditional_probability)
        for j in range(1, 10):
            probability = cal_probability(img, j, prior_probability, conditional_probability)
            if max_probability < probability:
                max_probability = probability
                max_label = j
        predict_y.append(max_label)  # 用来记录每行最大概率的label
    return np.array(predict_y)


def cal_accuracy(test_y, predict_y):
    m = test_y.shape[0]
    errorCount = 0.0
    for i in range(m):
        if test_y[i] != predict_y[i]:
            errorCount += 1
    accuracy = 1.0 - float(errorCount) / m
    return accuracy


if __name__ == '__main__':
    classNum = 10
    time1 = time.time()
    print('read data =====================')
    train_x, test_x, train_y, test_y = loadDataSet()
    time2 = time.time()


    print('train model =====================')
    prior_probability, conditional_probability = train_model(train_x, train_y, classNum)
    # for i in tqdm(range(classNum)):
    #     print(prior_probability[i])  # 输出一下每个标签的总共数量
    time3 = time.time()
    print("train model cost", time3 - time2, "seconds\n")

    print('predict classification ======================')
    predict_y = predict(test_x, test_y, prior_probability, conditional_probability)
    time4 = time.time()
    print("predict classification", time4 - time3, "seconds\n")

    acc = cal_accuracy(test_y, predict_y)
    time5 = time.time()
    print("accuarcy", acc)