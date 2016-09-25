# -*- coding:utf-8 -*-
from math import log
import operator

u"""
    使用ID3算法划分决策树（？）

    利用字典结构实现树 其中键是分类特征 值是分类名称
    叶子节点的值是字符串之类的 其他节点是字典
    通过递归的方式构成整个树
 """


def createDataSet():
    u"""
     创建测试数据 是不是鱼的那个
     """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calShannonEnt(dataSet, index=-1):
    u"""
     计算数据集信息熵 H = -sum(p(x) * log(p(x), 2))

     参数：
     dataSet 数据集
     index 待计算标签序号 默认为最后一列
     """
    numEntries = len(dataSet)
    labelCounts = {}
    # 为所有可能的分类创建字典 键为分类标签 值为出现次数
    for featVec in dataSet:
        currrentLabels = featVec[index]
        if currrentLabels not in labelCounts.keys():
            labelCounts[currrentLabels] = 0
        labelCounts[currrentLabels] += 1
    shannonEnt = 0.0
    # 计算信息熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    u"""
     划分数据集 若测试特征一致返回其他特征
     （不包括测试特征）

     参数：
     dataSet 待划分的数据集
     axis 划分数据集的特征所在的序号
     value 需要返回的特征的值
     """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    u"""
     选择最好的数据划分方式

     参数：
     dataSet 待选择数据集

     遍历当前特征中的所有唯一属性值，对每一特征划分一次数据集，
     然后计算数据集的新熵，并对所有唯一特征值得到的熵求和，最
     后比较所有特征中的信息增益（熵的减少/无序度的减少），返
     回最好特征（信息增益最大）的索引值
     """
    numFeatures = len(dataSet[0]) - 1
    bestEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 利用列表推导遍历特征值
        featList = [example[i] for example in dataSet]
        # 利用集合去重 保证唯一最快实现方式
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # 加权求和
            newEntropy += prob * calShannonEnt(subDataSet)
            # 信息增益
        infoGain = bestEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    u"""
     采用多数表决法决定叶子节点分类

     参数：
     classList 分类名称列表

     以字典形式储存标签及其出现频率 利用operator操作键值排序字典
     返回出现次数最多的分类名称
     """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        sortedClassCount = sorted(
            classCount.iteritems(),
            key=operator.itemgetter(1),
            reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    u"""
     创建决策树

     参数：
     dataSet 数据集
     labels 标签

     得到原始数据集，基于最好的属性值划分数据集，由于特征值可能多于两个，
     因此可能出现两个以上的数据集分支。第一次分割后，数据将向下传递给树
     分支的下一个节点递归的分析。直到：1.所有类标签相同；2.特征已用完
    """
    classList = [example[-1] for example in dataSet]
    # 类别相同时停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时 返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最好的特征划分
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 节点标签
    bestFeatLabels = labels[bestFeat]
    # 以字典结构存储树（可以优化为其他数据结构？）
    myTree = {bestFeatLabels: {}}
    # 删除已使用标签
    del(labels[bestFeat])
    # 对剩余数据递归分析 分析剩余特征并划分遍历
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabels][value] = \
            createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec, classLabel=None):
    u"""
     使用决策树执行分类

     参数：
     inputTree 字典形式的决策树
     featLabels 分类标签
     testVec 测试数据

     """
    # 解析树 firstStr是节点标签 secondDict是其子树（叶子节点的不是字典）
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # 将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            # 对子树递归
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(
                    secondDict[key], featLabels, testVec, classLabel)
            # 达到叶子节点则返回叶子节点标签
            else:
                classLabel = secondDict[key]
        # BUG：万一没找到怎么办（？）修复：即classLabel未声明直接返回了 catch到返回一个字符串
    if classLabel:
        return classLabel
    else:
        return 'Error : Not Found!'


def storeTree(inputTree, filename):
    u"""
     利用pickle序列化对象 保存决策树

     参数：
     input 决策树
     filename 目标文件名
     """
    import pickle
    with open(filename, 'w') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    u"""
     利用pickle序列化对象 读取决策树

     参数：
     filename 保存文件名
     """
    import pickle
    with open(filename) as fr:
        # 这个写法我觉着多少有些问题 离开这个函数时对象fr就会被销毁吧
        return pickle.load(fr)
    #     outputTree = pickle.load(fr)
    # return outputTree
