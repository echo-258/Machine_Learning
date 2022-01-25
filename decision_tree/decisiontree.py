import math
import csv
import time

class Node():
    def __init__(self, child=None, shannon=None, label=None, feature=None, set=None):
        self.child = child
        self.shannon = shannon
        self.label = label
        self.feature = feature
        self.set = set


class Tree():
    def __init__(self, dataSet, featSet):
        self.dataSet = dataSet
        self.featSet = featSet
        self.head = None

    def calculShannon(self, dataSet):       # 熵的基本计算公式
        num = {}
        for example in dataSet:
            num[example[-1]] = num.setdefault(example[-1], 0) + 1
        all_num = len(dataSet)
        shannon = 0.0
        for label in num:
            shannon -= num[label] / all_num * math.log2(num[label] / all_num)
        return shannon

    def splitDataSet(self, dataSet, index, value):  # 根据选得的最优特征划分数据集
        cutDataSet = []
        for example in dataSet:
            if example[index] == value:
                cutDataSet.append(example[:index] + example[index + 1:])
        return cutDataSet

    def majorLabel(self, dataSet):  # 无法得到精确划分时，选取剩余数据中占多数的样本作为决策依据
        num = {}
        for example in dataSet:
            num[example[-1]] = num.setdefault(example[-1], 0) + 1
        return sorted(num.items(), key=lambda x: x[1], reverse=True)[0][0]

    def chooseBestFeature(self, dataSet):
        bestFeature = None
        bestShannon = 0.0
        shannon = self.calculShannon(dataSet)   # 选取每层最优特征的依据：计算经验条件熵、信息增益
        all_num = len(dataSet)
        for index in range(len(dataSet[0]) - 1):
            featList = [example[index] for example in dataSet]
            featList = set(featList)
            curShannon = 0
            if len(featList) == 1:
                continue
            else:
                for featValue in featList:
                    cutDataSet = self.splitDataSet(dataSet, index, featValue)
                    curShannon += len(cutDataSet) / all_num * self.calculShannon(cutDataSet) # 计算经验条件熵
                infoGain = shannon - curShannon     # 计算信息增益
                if infoGain > bestShannon:
                    bestShannon = infoGain
                    bestFeature = index
        return bestFeature

    def createTree(self):
        def create(dataSet, featSet):
            if len(self.dataSet[0]) == 1:   # 划分至只有一个数据
                label = self.majorLabel(dataSet)
                node = Node(label=label, set=dataSet)
                return node

            labelList = [example[-1] for example in dataSet]
            labelList = list(set(labelList))
            if len(labelList) == 1:     # 只剩一种备选特征
                node = Node(label=labelList[0], set=dataSet)
                return node
            bestFeature = self.chooseBestFeature(dataSet)
            if bestFeature == None:     # 若有多种划分选择，则寻找最优划分特征
                label = self.majorLabel(dataSet)
                node = Node(label=label, set=dataSet)   # 找不到则按照多数样本点的label作为结点
                return node
            bestFeatureValues = [example[bestFeature] for example in dataSet]
            bestFeatureValues = set(bestFeatureValues)  # 找到最优特征，则根据此特征的不同取值划分不同子树
            node = Node(child={}, feature=bestFeature, set=dataSet)
            del featSet[bestFeature]
            for value in bestFeatureValues:
                cutDataSet = self.splitDataSet(dataSet, bestFeature, value)
                node.child[value] = create(cutDataSet, featSet[:])
            return node

        dataSet = self.dataSet[:]
        featSet = self.featSet[:]
        node = create(dataSet, featSet)     # 递归开始
        self.head = node


    def Predict(self, testData):    # 决策树模型测试
        def predict(test, node):
            if node.child == None and node.label != None:   # 递归到底层
                return node.label
            index = node.feature
            cur = test[index]
            if cur not in node.child.keys():
                return self.majorLabel(node.set)
            else:
                del test[index]     # 从待测特征列表中删去此特征
                return predict(test, node.child[cur])   # 在决策树中，依照样本特征递归搜索

        correct = 0
        testNum = len(testData)
        for test in testData:
            res = predict(test[:], self.head)
            if res == test[-1]:
                correct += 1
        # print(correct, testNum)
        print('accuracy: ', correct / testNum)

def read_data(path):
    fplist = []
    num = 0
    with open(path, 'r') as fp:
        content = csv.reader(fp)
        for row in content:
            if num == 0:
                featureSet = row
                num = 1
                continue
            else:
                fplist.append(list(map(int, row)))
    featureSet.pop(-1)
    allNum = len(fplist)
    trainNum = int(allNum * 0.1)   # 划分训练集与测试集
    trainDataSet = fplist[:trainNum]
    testDataSet = fplist[trainNum:]
    return trainDataSet, testDataSet, featureSet

if __name__ == '__main__':
    time1 = time.time()
    print('read data set =================')
    trainSet, testSet, featureSet = read_data('train.csv')
    time2 = time.time()
    print('read data cost', time2 - time1, 'seconds\n')

    time3 = time.time()
    print('train model ===================')
    tree = Tree(trainSet, featureSet)
    tree.createTree()
    time4 = time.time()
    print('train model cost', time4 - time3, 'seconds\n')
    # tree.show()

    print('test classification ==================')
    tree.Predict(testSet)