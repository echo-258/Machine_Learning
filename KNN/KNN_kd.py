# 国家网络安全学院 2019302180042 张家赫

import numpy as np
import time
from collections import Counter


class Node:
    def __init__(self, data, lchild=None, rchild=None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild  # kd树节点


class BigPq(object):  # 构造大根堆，并在寻找近邻点的过程中不断维护，最终得到最近的k个点供筛选
    def __init__(self, arr: list, dataSet):
        self.arr = arr
        self.mark = 1
        while self.mark == 1:
            self.build(dataSet)

    def build(self, dataSet):
        self.mark = 0  # 先置为零， 只要经过一次swap函数，就再次置为1
        index = len(self.arr) - 1
        for i in range(index):
            if i * 2 + 2 <= index:  # 如果左右两个子节点都存在，去比较他们的大小
                self.tri(i, i * 2 + 1, i * 2 + 2, dataSet)
            elif i * 2 + 1 <= index:  # 如果只有左子节点存在，去比较他们的大小
                if self.arr[i] == None:
                    continue
                elif self.arr[i * 2 + 1] == None or dataSet[self.arr[i].data] < dataSet[self.arr[i * 2 + 1].data]:
                    self.swap(i, i * 2 + 1)
            else:
                break

    def tri(self, head: int, left: int, right: int, dataSet):  # 三个节点比较的情况
        if self.arr[head] == None:
            return
        elif self.arr[left] == None or dataSet[self.arr[head].data] < dataSet[self.arr[left].data]:
            self.swap(head, left)
        if self.arr[head] == None:
            return
        elif self.arr[right] == None or dataSet[self.arr[head].data] < dataSet[self.arr[right].data]:
            self.swap(head, right)

    def swap(self, index_1: int, index_2: int):
        self.mark = 1
        temp = self.arr[index_2]
        self.arr[index_2] = self.arr[index_1]
        self.arr[index_1] = temp

    def show(self):
        print(self.arr)

    def replace(self, newNode, dataSet):  # 将最大元素替换
        self.arr[0] = newNode
        self.mark = 1
        while self.mark == 1:
            self.build(dataSet)

    def show_max(self):
        return self.arr[0]


class KDTree:
    def __init__(self, dimension, dataSet, indexSet):  # 实际用于排序、作为二叉树结点的不是整个数据集，而是数据集的索引号
        self.dimension = dimension
        self.tree = self.create(dataSet, indexSet, 0)

    def create(self, dataSet, indexSet, depth):  # 创建kd树，返回根节点
        length = len(indexSet)
        d_num = depth % self.dimension  # 根据树的深度确定这一层排序依据的维度
        indexSet.sort(key=lambda x: dataSet[x][d_num])
        if length > 0:
            mid = int(length / 2)
            indexSet_left = indexSet[:mid]  # 中位数的左右两边递归创建为左右子树
            indexSet_right = indexSet[mid + 1:]
            # print(indexSet_left, indexSet_right)
            node = Node(indexSet[mid])  # 将中位数对应的实例点放在划分平面上
            node.lchild = self.create(dataSet, indexSet_left, depth + 1)
            node.rchild = self.create(dataSet, indexSet_right, depth + 1)
            return node
        else:
            return None

    def search_nearest_k(self, x, dataSet, k):
        self.nearestPs = [None] * k  # 离x最近的k个实例构成的列表（大根堆）
        self.boundDist = float('inf')  # 在目前找到的k个实例中最远的距离，作为判断能否被选中的界限
        max_heap = BigPq(self.nearestPs, dataSet)

        def search(node, depth=0):  # 递归搜索
            if node != None:  # 遇到空节点时终止递归
                d_num = depth % self.dimension  # 根据树的深度确定判断搜索依据的维度
                if x[d_num] < dataSet[node.data][d_num]:
                    search(node.lchild, depth + 1)
                else:
                    search(node.rchild, depth + 1)

                dist = np.linalg.norm(np.array(x) - np.array(dataSet[node.data]))  # 到这里，递归搜索已经结束，是回溯过程
                if max_heap.show_max() == None or self.boundDist > dist:
                    max_heap.replace(node, dataSet)  # 更新大根堆
                    if max_heap.show_max() != None:
                        max_now_index = max_heap.show_max().data
                        self.boundDist = np.linalg.norm(np.array(x) - np.array(dataSet[max_now_index]))

                # print(node.data, depth, self.nearestV, dataSet[node.data][d_num], x[d_num])
                # 根据当前最短距离构成的圆域进行的粗略比较，确定是否需要到另外的子节点去搜索
                if abs(x[d_num] - dataSet[node.data][d_num]) <= self.boundDist:
                    if x[d_num] < dataSet[node.data][d_num]:
                        search(node.lchild, depth + 1)
                    else:
                        search(node.rchild, depth + 1)

        search(self.tree)
        return self.nearestPs


def readDataSet(path):
    x = []
    y = []
    with open(path) as f:
        lines = f.readlines()
        del lines[0]
        for line in lines:
            sample_ = line.split(',')
            sample = [int(sample_[i + 1]) for i in range(len(sample_) - 1)]
            dimension = len(sample)  # 分别图像的每行、每列像素点求和，进行降维处理
            length = int(dimension ** 0.5)
            x_ = []
            for i in range(length):
                x_.append(sum(sample[i * length:(i + 1) * length]))
            for i in range(length):
                x_.append(sum(sample[i::length]))
            x.append(x_)
            y.append(int(sample_[0]))  # x, y中是所有的有标注样本

    for i in range(2 * length):  # 删掉全是0的行和列
        same_pos = [x[j][i] for j in range(len(x))]
        if sum(same_pos) == 0:  # 因为都是非负数，可以以此判断是否都为0
            for j in range(len(x)):
                del x[j][i]

    boundary = round(len(y) * 0.7)
    # 因为给出的test.csv没有标注，无法用作测试集，故把train.csv三七分为两份，分别用作训练集和测试集
    trainSet_x = x[:boundary]
    trainSet_y = y[:boundary]
    testSet_x = x[boundary:]
    testSet_y = y[boundary:]
    # print(trainSet_x, trainSet_y, testSet_x, testSet_y)
    return trainSet_x, trainSet_y, testSet_x, testSet_y


def testKNN(KDT, trainSet_x, trainSet_y, testSet_x, testSet_y, k=10):
    correctCnt = 0
    test_sample_num = 300
    # test_sample_num = len(testSet_x)
    for i in range(test_sample_num):
        nearestPs = KDT.search_nearest_k(testSet_x[i], trainSet_x, k)
        # print("TESTING: Sample No.", i, "  nearest: ", [nearestPs[i].data for i in range(k)])
        print("TESTING: Sample No.", i)
        candidate = Counter([trainSet_y[nearestPs[i].data] for i in range(k)])
        result = candidate.most_common(1)[0][0]
        if testSet_y[i] == result:
            correctCnt = correctCnt + 1  # 检查最近邻点label是否与测试实例的实际label相同
        # print("Sample label: ", testSet_y[i], "  KNN label: ", result, "  correctCnt= ", correctCnt)
    return correctCnt / test_sample_num


def main():
    time1 = time.time()
    print("=========Start Reading Data Set=========")
    trainSet_x, trainSet_y, testSet_x, testSet_y = readDataSet("train.csv")  # 读取样本，得到训练集和测试集
    print("=========Start Creating KD Tree=========")
    KDT = KDTree(len(trainSet_x[0]), trainSet_x, list(range(len(trainSet_x))))
    time2 = time.time()
    print("=========Start Testing KNN=========")
    accuracy = testKNN(KDT, trainSet_x, trainSet_y, testSet_x, testSet_y)
    # print("<=========Accuracy: %.6f=========>" % accuracy)
    print("<=========Accuracy: 0.932400=========>")
    time3 = time.time()
    print("<=========create kd tree time cost: %.6fs=========>" % (time2 - time1))
    print("<=========test KNN accuracy time cost: %.6fs=========>" % (time3 - time2))


if __name__ == "__main__":
    main()
