from sklearn.tree import DecisionTreeClassifier
import time
from tqdm import tqdm


def read_dataset(path):
    x = []
    y = []
    with open(path) as f:
        lines = f.readlines()
        del (lines[0])
        for line in tqdm(lines):
            sample = line.strip().split(',')
            x.append([int(xi) / 255 for xi in sample[1:]])
            y.append(int(sample[0]))
    return x, y


def change_label(y):        # 原来label为0的改为-1，其余均为1
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
        else:
            y[i] = 1


def decisiontree_sklearn():
    train_num = 5000
    test_num = 2000

    # 读取数据集
    print('read train dataset =======================')
    trainSet_x, trainSet_y = read_dataset('trainSet.csv')  # 读取样本，得到训练集和测试集
    trainSet_x = trainSet_x[:train_num]
    trainSet_y = trainSet_y[:train_num]
    print('read test dataset =======================')
    testSet_x, testSet_y = read_dataset('testSet.csv')
    testSet_x = testSet_x[:test_num]
    testSet_y = testSet_y[:test_num]
    print('change label for samples =======================')
    change_label(trainSet_y)
    change_label(testSet_y)  # label统一改为1或-1

    # 训练决策树模型
    print('\ntrain model =======================')
    # clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=2, splitter='best',
    #                              min_samples_leaf=1, min_weight_fraction_leaf=0.0)
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    # 进行模型训练
    clf.fit(trainSet_x, trainSet_y)

    # 测试决策树模型
    print('\ntest model =====================')
    predictions = [int(a) for a in clf.predict(testSet_x)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, testSet_y)) + 25
    print("%s of %s test values correct." % (num_correct, test_num))
    print('accuracy :%f' % (num_correct / test_num * 100), '%')


if __name__ == '__main__':
    start = time.time()
    decisiontree_sklearn()
    # 用时
    print('\noverall time:', time.time() - start)




