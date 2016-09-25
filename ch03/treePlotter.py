# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def retrieveTree(i):
    u"""
     预存储的树的信息 测试用

     参数：
     i 预存储树的序号
     """
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers':{0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]


def getNumLeafs(myTree):
    u"""
     获取叶节点数目

     参数：
     myTree 树

     这里使用字典结构存储树 也就是当前遍历的节点如果
     不是字典 则该节点是一个叶子节点 统计的叶子节点
     数+1 否则不是叶子节点 递归调用该函数
     """
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    u"""
     获取树的深度

     参数：
     myTree 树

     这里使用字典结构存储树 也就是当前遍历的节点如果
     不是字典 则该节点是一个叶子节点 当前子树深度+1
     否则不是叶子节点 递归调用该函数 在循环结束前比较
     当前子树深度与最大深度 取其较大者
     """
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    u"""
     绘制节点

     参数：
     nodeTxt 节点名称（？）
     centerPt 起点和终点中的某一个
     parentPt 起点和终点中的某一个
     nodeType 节点外观 预定义内容

     实质是注解工具annotates
     """
    createPlot.ax1.annotate(
        nodeTxt, xy=parentPt, xycoords='axes fraction',
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType,
        arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    u"""
     在父子节点中间（几何中点）添加文本信息

     参数：
     cntrPt 起点和终点中的某一个
     parentPt 起点和终点中的某一个
     txtString 标注信息

     用于标记子节点属性值
     """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    u"""
     绘制子树

     参数：
     myTree 子树
     parentPt 当前节点坐标（？）
     nodeTxt 当前节点属性（？）

     参考函数内注释
     其中：
     plotTree.xOff 追踪已绘制节点位置
     plotTree.yOff 追踪已绘制节点位置
     plotTree.totalW 整个树的宽度（叶子节点数）
     plotTree.totalD 整个树的高度
     """
    # 计算叶子节点数量与树高度
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    # 没太弄明白 大概就是绘制当前节点 连接当前节点与其父节点并且标注当前节点属性
    cntrPt = (
        plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
        plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # 自顶向下绘制 因此减少y偏移
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # 递归绘制子树
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        # 绘制叶子节点
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(
                secondDict[key],
                (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    u"""
     绘制的主函数

     参数：
     inTree 待绘制的树

     参考函数内注释
     其中：
     plotTree.xOff 追踪已绘制节点位置
     plotTree.yOff 追踪已绘制节点位置
     plotTree.totalW 整个树的宽度（叶子节点数）
     plotTree.totalD 整个树的高度
     """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
