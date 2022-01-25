import warnings
import numpy as np
import time
import random
from tqdm import tqdm
warnings.filterwarnings('ignore')


def readDataSet(path):
    x = []
    y = []
    with open(path) as f:
        lines = f.readlines()
        del (lines[0])
        for line in lines:
            sample = line.split(',')
            x_ = []
            for i in sample[1:]:
                x_.append(int(i))
            x.append(x_)
            y.append(int(sample[0]))  # x, y中是所有的有标注样本
    # boundary = int(len(y) * 0.7)
    # # 因为给出的test.csv没有标注，无法用作测试集，故把train.csv三七分为两份，分别用作训练集和测试集
    trainSet_x = x[:5000]
    trainSet_y = y[:5000]
    testSet_x = x[5001:6000]
    testSet_y = y[5001:6000]
    # print(trainSet_x, trainSet_y, testSet_x, testSet_y)
    return trainSet_x, trainSet_y, testSet_x, testSet_y

def change_label(y):
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = 0
        else:
            y[i] = 1
    # 原来label为0的改为0，其余均为1


def Logistic(data, label, itertime = 1000):
    # 初始化w
    FeatureNum = len(data[0])
    w = np.zeros(FeatureNum)
    h = 0.0001       # 设定学习率

    # 样本数
    SampleNum = len(data)
    # 将data转换成ndarray格式，方便后续的计算
    data = np.array(data)

    # 迭代
    for i in tqdm(range(itertime)):
        # 随机选择误分类点
        flag = 0
        while flag == 0:
            # 随机选择
            s = random.sample(range(0, SampleNum - 1), 1)[0]
            xi = data[s]
            yi = label[s]

            # 如果分类错误，更新w
            if predict(w, xi) != yi:
                exp_wxi = np.exp(np.dot(w, xi))
                w += h * (xi * yi - (xi * exp_wxi) / (1 + exp_wxi))
                flag = 1
    return w



def predict(w, x):
    exp_wx = np.exp(np.dot(w, x))
    P = exp_wx / (1 + exp_wx)
    if P > 0.5:
        return 1
    return 0

def Classifier(data, label, w):
    # 样本数
    SampleNum = len(data)
    # 初始化错误分类的数量
    errorCnt = 0

    # 遍历每一个样本
    for i in range(SampleNum):

        # 对该样本的分类
        result = predict(w, data[i])

        # 判断是否分类正确
        if result != label[i]:
            # 分类错误，errorCnt+1
            errorCnt += 1

    # 计算正确率
    Acc = 1 - errorCnt / SampleNum
    return Acc


if __name__ == "__main__":
    time1 = time.time()
    print('read data set =================')
    train_data, train_label, test_data, test_label = readDataSet("train.csv")  # 读取样本，得到训练集和测试集
    change_label(train_label)
    change_label(test_label)  # label统一改为0或-1
    time2 = time.time()
    print('read data cost:', time2 - time1, 'seconds.\n')

    # 最优化参数
    print('train model =====================')
    time3 = time.time()
    w = Logistic(train_data, train_label)
    print('end training')
    time4 = time.time()
    print('training time: ', time4 - time3, 'seconds.\n')

    # 进行分类
    print('test classification =======================')
    time5 = time.time()
    accuracy = Classifier(test_data, test_label, w)
    time6 = time.time()
    print('testing time: ', time6 - time5, 'seconds.\n')
    print('accuracy: ', accuracy)

