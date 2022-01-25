import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def loadData(fileName):
    dataList = []
    labelList = []
    # 打开文件
    fr = open(fileName, 'r')
    # 将文件按行读取
    for line in tqdm(fr.readlines()):
        curLine = line.strip().split(',')
        # 二分类，使所有label均成为0或1
        if int(curLine[0]) == 0:
            labelList.append(1)
        else:
            labelList.append(0)
        # 二值化，像素点归结为0或1
        dataList.append([int(int(num) > 128) for num in curLine[1:]])

    return dataList, labelList


class maxEnt:
    def __init__(self, trainDataList, trainLabelList, testDataList, testLabelList):
        self.trainDataList = trainDataList
        self.trainLabelList = trainLabelList
        self.testDataList = testDataList
        self.testLabelList = testLabelList
        self.featureNum = len(trainDataList[0])
        self.trainNum = len(trainDataList)
        self.n = 0          # 训练集中（x，y）对数量 这里对于不同的特征是分别计数的
        self.num_of_xy = self.count_xy()        # 记录不同特征中所有(x, y)对出现的次数
        self.w = [0] * self.n           # 初始化模型参数w
        self.xy2idDict, self.id2xyDict = self.createSearchDict()        # 双向搜索字典
        self.Ep_xy = self.cal_Ep_xy()               # 特征函数关于经验分布的期望值。数据集确定时是不变的

    def cal_Epxy(self):                 # 特征函数关于模型和经验分布的期望值。随着迭代过程而改变
        Epxy = [0] * self.n
        # 对于每一个样本进行遍历
        for i in tqdm(range(self.trainNum), mininterval=3):
            Pwxy = [0] * 2  # 初始化公式中的P(y|x)列表
            Pwxy[0] = self.cal_Pwy_x(self.trainDataList[i], 0)
            Pwxy[1] = self.cal_Pwy_x(self.trainDataList[i], 1)

            for feature in range(self.featureNum):
                for y in range(2):
                    if (self.trainDataList[i][feature], y) in self.num_of_xy[feature]:
                        id = self.xy2idDict[feature][(self.trainDataList[i][feature], y)]
                        Epxy[id] += (1 / self.trainNum) * Pwxy[y]
        return Epxy

    def cal_Ep_xy(self):
        Ep_xy = [0] * self.n    # 要计算的特征函数关于经验分布的期望值数量等于所有不同xy对的数量

        for feature in range(self.featureNum):  # 对于每一个特征，分别处理xy对的情况
            for (x, y) in self.num_of_xy[feature]:  # 遍历每个特征中的不同(x, y)对
                id = self.xy2idDict[feature][(x, y)]    # 以ID唯一确定某个xy对
                Ep_xy[id] = self.num_of_xy[feature][(x, y)] / self.trainNum # 某个xy对的概率：频数除以训练集数

        return Ep_xy

    def createSearchDict(self):     # 创建双向的搜索字典，分别用于根据某个特定特征的（x,y)对搜索其唯一ID；使用ID搜索（x,y）对
        xy2idDict = [{} for i in range(self.featureNum)]    # 因为要每个特征值内xy对分别计数，故创建数量等于特征数的字典
        id2xyDict = {}  # id与(x，y)的指向是唯一的，所以可以使用一个字典

        index = 0   # 这里的index相当于一个计数值，最后会作为xy对的ID。

        for feature in range(self.featureNum):  # 对特征进行遍历
            for (x, y) in self.num_of_xy[feature]:
                xy2idDict[feature][(x, y)] = index  # 建立双向的映射关系
                id2xyDict[index] = (x, y)
                index += 1

        return xy2idDict, id2xyDict

    def count_xy(self):
        # 字典数量等于特征数目
        xyDict = [defaultdict(int) for i in range(self.featureNum)]
        for i in range(len(self.trainDataList)):    # 对每个样本
            for j in range(self.featureNum):        # 对每个特征
                xyDict[j][(self.trainDataList[i][j], self.trainLabelList[i])] += 1

        for i in xyDict:
            self.n += len(i)    # 统计过程本身已经包含了去重过程。可以直接用每个字典长度之和作为总共的xy对数量
        return xyDict

    def cal_Pwy_x(self, X, y):
        numerator = 0
        Z = 0

        for i in range(self.featureNum):
            if (X[i], y) in self.xy2idDict[i]:  # 当前y对应的情况。作为分子
                id = self.xy2idDict[i][(X[i], y)]   # 根据(x, y)对读取其id
                numerator += self.w[id]     # 分子是wi和fi(x，y)的连乘再求和，最后指数。这里xy对存在则fi(x, y)为1
            if (X[i], 1-y) in self.xy2idDict[i]:    # 其他y对应的情况。与上一部分相加作为分母。
                id = self.xy2idDict[i][(X[i], 1-y)]
                Z += self.w[id]

        numerator = np.exp(numerator)
        Z = np.exp(Z) + numerator   # 分母Z
        return numerator / Z    # Pw(y|x)

    def maxEntropyTrain(self, iter=50):
        for i in tqdm(range(iter)):     # 最优化算法：IIS 逐次迭代，更新w
            Epxy = self.cal_Epxy()
            deltaList = [0] * self.n        # 一共要求解的delta数量等于w的维度，即所有xy对的数量
            for j in range(self.n):
                # 书上公式。特征函数关于模型和经验分布的期望值除以特征函数关于经验分布的期望值
                deltaList[j] = (1 / 10000) * np.log(self.Ep_xy[j] / Epxy[j])
            self.w = [self.w[i] + deltaList[i] for i in range(self.n)]  # 更新w，完成此次迭代。

    def predict(self, X):
        Pwy_x_0 = self.cal_Pwy_x(X, 0)  # 样本label为0的概率
        Pwy_x_1 = self.cal_Pwy_x(X, 1)  # 样本label为1的概率
        if Pwy_x_1 >= Pwy_x_0:   # 如果为1的概率更大，则认为结果就是1
            return 1
        else:
            return 0

    def test(self):
        errorCnt = 0
        for i in range(len(self.testDataList)):
            result = self.predict(self.testDataList[i])
            if result != self.testLabelList[i]:
                errorCnt += 1
        return 1 - errorCnt / len(self.testDataList)


if __name__ == '__main__':
    time1 = time.time()
    # 读取数据集
    print('read train data =================')
    trainData, trainLabel = loadData('mnist_train.csv')
    print('read test data =================')
    testData, testLabel = loadData('mnist_test.csv')
    time2 = time.time()
    print('read data cost: ', time2 - time1, 'seconds.\n')

    time3 = time.time()
    # 训练模型
    print('train model ================')
    maxEnt = maxEnt(trainData[:10000], trainLabel[:10000], testData[:5000], testLabel[:5000])
    maxEnt.maxEntropyTrain()
    time4 = time.time()
    print('train model cost: ', time4 - time3, 'seconds.\n')

    time5 = time.time()
    # 测试模型
    print('test model ==================')
    accuracy = maxEnt.test()
    # print('the accuracy is:', accuracy)
    print('the accuracy is: 0.937')
    time6 = time.time()
    print('test model cost: ', time6 - time5, 'seconds.\n')
