# -*- coding:utf-8 -*-

from numpy import *
import operator


def classify0(inX, dataSet, labels, k):
    u'''
     利用kNN算法进行分类

     参数：
     inX 待分类向量
     dataSet 训练样本向量
     labels 标签向量
     k 选择最近邻居数量

     计算待分类向量与训练样本向量的欧氏距离，
     选取距离最近的k个点，返回k个点中出现频率最高
     的点的标签作为预测结果
     '''
    dataSetSize = dataSet.shape[0]
    # 计算欧几里得距离  但是我没明白这里具体的计算方式 和numpy有关吧
    # np.tile(a,rep) 将a按结构重复rep次 
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
    # 排序 按照第二个元素（字典的值，也就是出现次数）逆序排列
    sortedClassCount = sorted(classCount.iteritems(),
         key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的标签的值
    return sortedClassCount[0][0]


def autoNorm(dataSet):
    u"""
     数据归一化

     参数：
     dataSet 待归一化数据集

     normValue = (oldValue-min)/(max-min)
     """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSets = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSets = dataSet - tile(minVals, (m, 1))
    normDataSets = normDataSets / tile(ranges, (m, 1))
    return normDataSets, ranges, minVals
