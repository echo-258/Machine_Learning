import numpy as np
import time


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
    boundary = int(len(y) * 0.7)
    # 因为给出的test.csv没有标注，无法用作测试集，故把train.csv三七分为两份，分别用作训练集和测试集
    trainSet_x = x[:boundary]
    trainSet_y = y[:boundary]
    testSet_x = x[boundary:]
    testSet_y = y[boundary:]
    # print(trainSet_x, trainSet_y, testSet_x, testSet_y)
    return trainSet_x, trainSet_y, testSet_x, testSet_y

def function(_list, index):
    _list[index] = '-1'


def change_label(y):
    for i in range(len(y)):
        if y[i] == 0:
            # function(y, i)
            y[i] = -1
        else:
            y[i] = 1
    # 原来label为0的改为-1，其余均为1


def train_perceptron(trainSet_x, trainSet_y, learn_rate=0.9, iter = 30):
    iteration = 0
    m = len(trainSet_x)
    n = len(trainSet_x[0])
    w = np.zeros((1, n))
    b = 0
    while iteration < iter:  # 无差错分类基本不可能达到，故以迭代次数达到一定值作为迭代终止条件
        for i in range(m):
            # 每次选取样本点的方法与课本上有所不同，是一轮一轮迭代进行，每一轮依次选取每个样本点一次，使得所有样本点对模型建立的贡献相当
            xi = np.array(trainSet_x[i])
            if ((w.dot(xi.T) + b) * trainSet_y[i]) <= 0:
                # 采用感知机原始形式
                w = w + learn_rate * trainSet_y[i] * xi
                b = b + learn_rate * trainSet_y[i]
        iteration = iteration + 1
        print("iteration: ", iteration)
    return w, b


def testPerceptron(w, b, testSet_x, testSet_y):
    m = len(testSet_x)
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(m):
        xi = np.array(testSet_x[i])
        if testSet_y[i] == 1:  # 正类
            if (w.dot(xi.T) + b) > 0:
                TP = TP + 1  # 将正类预测为正类
            else:
                FN = FN + 1  # 将正类预测为负类
        else:  # 负类
            if (w.dot(xi.T) + b) > 0:
                FP = FP + 1  # 将负类预测为正类
            else:
                TN = TN + 1  # 将负类预测为负类
    # print("TP=", TP, " FN=", FN, " FP=", FP, " TN=", TN, " m=", m)
    return (TP + TN) / m, TP / (TP + FP), TP / (TP + FN)


def main():
    start_t = time.time()
    trainSet_x, trainSet_y, testSet_x, testSet_y = readDataSet("train.csv")  # 读取样本，得到训练集和测试集
    change_label(trainSet_y)
    change_label(testSet_y)  # label统一改为1或-1
    w, b = train_perceptron(trainSet_x, trainSet_y)
    # 训练模型
    print("\nw= ", w, "b= ", b)
    accuracy, precision, recall = testPerceptron(w, b, testSet_x, testSet_y)  # 测试模型
    print("\naccuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)  # 输出结果
    end_t = time.time()
    print("time cost: ", end_t - start_t, "s")


if __name__ == "__main__":
    main()
