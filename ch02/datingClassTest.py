# -*- coding:utf-8 -*-

from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return (group, labels)


def classify0(inX, dataSet, labels, k):
    u'''
     输入参数共有四个：用于分类的输入向量inX,输入的训练样本集dataSet,
     标签向量labels,用于选择最近邻居数目的参数k，其中练样本集dataSet
     的行数和标签向量labels的元素数目相同
     '''
    dataSetSize = dataSet.shape[0]
    # 计算距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    #  numpy.argsort() 按照从小到大的顺序返回被排序元素的index
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.iteritems(),\
         key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    with open(filename) as fr:
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0: 3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSets = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSets = dataSet - tile(minVals, (m, 1))
    normDataSets = normDataSets / tile(ranges, (m, 1))
    return normDataSets, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs : m, :],\
            datingLabels[numTestVecs : m], 3)
        print "the classifier came back with: %d, the real answer is: %d" \
            % (classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1
    errorate = errorCount / float(numTestVecs) * 100
    print "the total error rate is: " + str(errorate) + "%"

