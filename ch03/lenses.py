# /usr/bin/env
# -*- coding:utf-8 -*-
import trees
import treePlotter


try:
    lensesTrees = trees.grabTree('lensesTrees.txt')
except:
    with open('lenses.txt') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLables = ['age', 'prescript', 'astigmatic', 'presbyopic']
    lensesTrees = trees.createTree(lenses, lensesLables)
    trees.storeTree(lensesTrees, 'lensesTrees.txt')
finally:
    test = ['pre', 'myope', 'yes', 'normal']
    lensesLables = ['age', 'prescript', 'astigmatic', 'presbyopic']
    result = trees.classify(lensesTrees, lensesLables, test)
    print result
    treePlotter.createPlot(lensesTrees)
