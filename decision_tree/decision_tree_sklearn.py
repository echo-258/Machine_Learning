import numpy as np
from sklearn import preprocessing
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
            x.append([int(xi) for xi in sample[1:]])
            y.append(int(sample[0]))
    return x, y

if __name__ == '__main__':
    start = time.time()
    # 获取训练集及标签
    print('读入训练数据集...')
    trainDataList, trainLabelList = read_dataset('trainSet.csv')
    # 获取测试集及标签
    print('读入测试数据集...')
    testDataList, testLabelList = read_dataset('testSet.csv')
    # 获取一个支持向量机模型
    predictor = DecisionTreeClassifier(criterion='entropy',min_samples_split=2,
    min_samples_leaf=1,min_weight_fraction_leaf=0.0)
    # 预测结果
    predictor.fit(trainDataList, trainLabelList)
    result = predictor.predict(testDataList)
    test_num = len(testDataList)
    # 开始测试
    print('模型测试中...')
    accuracy = np.sum(np.equal(result, testLabelList)) / test_num
    print('准确率 = %f' % accuracy)
    # 打印时间
    print('模型整体花费时间:', time.time() - start)