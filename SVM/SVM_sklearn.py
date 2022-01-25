import numpy as np
import time
from sklearn import svm
from sklearn import preprocessing
from tqdm import tqdm


def read_dataset(path):
    x = []
    y = []
    with open(path) as f:
        lines = f.readlines()
        del (lines[0])
        for line in tqdm(lines):
            sample = line.strip().split(',')
            x.append([int(xi) for xi in sample[1:]])
            y.append(int(sample[0]))
    return x, y


def change_label(y):        # 原来label为0的改为-1，其余均为1
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
        else:
            y[i] = 1


def SVM_sklearn():
    train_num = 2000
    test_num = 1000

    # 读取数据集
    print('read train dataset =======================')
    trainSet_x, trainSet_y = read_dataset('trainSet.csv')  # 读取样本，得到训练集和测试集
    trainSet_x = trainSet_x[:train_num]
    trainSet_y = trainSet_y[:train_num]
    print('read test dataset =======================')
    testSet_x, testSet_y = read_dataset('testSet.csv')
    testSet_x = np.array(testSet_x[:test_num])
    testSet_y = testSet_y[:test_num]
    print('\nrescale and change label for samples =======================')
    trainSet_x = preprocessing.StandardScaler().fit_transform(trainSet_x)
    testSet_x = preprocessing.StandardScaler().fit_transform(testSet_x)
    change_label(trainSet_y)
    change_label(testSet_y)  # label统一改为1或-1

    # 训练SVM模型
    print('\ntrain SVM =======================')
    # clf = svm.SVC(C=5000, kernel='rbf', gamma=1 / 1000000)
    clf = svm.SVC(kernel='rbf')
    # 进行模型训练
    clf.fit(trainSet_x, trainSet_y)

    # 测试SVM模型
    print('\ntest SVM =====================')
    predictions = [int(a) for a in clf.predict(testSet_x)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, testSet_y))
    print("%s of %s test values are correct." % (num_correct, len(testSet_y)))
    print('accuracy :%f' % (num_correct / len(testSet_y) * 100), '%')


if __name__ == "__main__":
    start = time.time()
    SVM_sklearn()
    # 用时
    print('\noverall time:', time.time() - start)
