#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""差分进化算法,针对单目标"""

import numpy as np
import DifferentialEvolution.DOE as DOE
from mayavi import mlab
import os
os.environ['QT_API']='pyside'


class Ind(object):

    def __init__(self,min=0,max=0,location=None):
        """个体生成采用两种方案。
        第一种，由进化算法采用DOE方法生成样本，通过Location来指定个体，此时min和max无效。
        第二种，location为None，则个体自己在采样空间【min，max】中随机生成"""
        if location is None:
            dim=min.shape[0]
            rand=np.random.uniform(0,1,dim)
            self.x=(max-min)*rand+min
        else:
            self.x=location

    def getValue(self,environment):
        self.y=environment(self.x)

    def __cmp__(self, other):
        if (self.y > other.y):
            return 1
        elif (self.y < other.y):
            return -1
        elif (self.y == other.y):
            return 0

    def __gt__(self, other):
        if (self.y > other.y):
            return True
        else:
            return False

    def __lt__(self, other):
        if (self.y < other.y):
            return True
        else:
            return False

    def __eq__(self, other):
        if (self.y == other.y):
            return True
        else:
            return False



class DE(object):
    def __init__(self,min,max,population,K,CR,environment,isMin=True):
        """min,max是一维数组，用以确定搜索空间的范围
        population是种群数量
        environment是函数指针，用以表示环境，检验个体的性能
        isMin用以确定是不是求最小值，默认为TRUE。若为FALSE表示求取最大值"""

        samples=DOE.LatinHypercube(dimension=min.shape[0],num=population,min=min,max=max)
        self.Inds=[]
        for i in range(population):
            ind=Ind(location=samples.realSamples[i,:])
            self.Inds.append(ind)
        self.environment=environment
        self.isMin=isMin
        self.K=K
        self.CR=CR

    def aberrance(self,K):
        """K为缩放比例因子"""
        nextInds=[]
        population=len(self.Inds)
        for i in range(population):
            isOK=True
            while isOK:
                r1=np.random.randint(0,population)
                if r1==i:
                    continue
                r2=np.random.randint(0,population)
                if r1==r2 or r2==i:
                    continue
                r3=np.random.randint(0,population)
                if r3==r1 or r3==r2 or r3==i:
                    continue
                isOK=False
            x=self.Inds[r1].x+K*(self.Inds[r2].x-self.Inds[r3].x)
            ind=Ind(location=x)
            nextInds.append(ind)
        self.nextInds=nextInds

    def exchange(self,CR):
        """CR是交叉因子"""
        inds1=self.Inds
        inds2=self.nextInds
        inds=[]
        population=len(inds1)
        dimension=inds1[0].x.shape[0]
        for i in range(population):
            x1=inds1[i].x
            x2=inds2[i].x
            x=np.zeros(dimension)
            randr = np.random.randint(0, dimension)
            for j in range(dimension):
                randb=np.random.uniform()
                if randb<=CR or j==randr:
                    x[j]=x2[j]
                else:
                    x[j]=x1[j]
            ind=Ind(location=x)
            inds.append(ind)
        self.nextInds=inds

    def select(self):
        inds=[]
        population=len(self.nextInds)
        for i in range(population):
            self.Inds[i].getValue(self.environment)
            self.nextInds[i].getValue(self.environment)
            if self.isMin:
                if self.nextInds[i].y>self.Inds[i].y:
                    inds.append(self.Inds[i])
                else:
                    inds.append(self.nextInds[i])
            else:
                if self.nextInds[i].y<self.Inds[i].y:
                    inds.append(self.Inds[i])
                else:
                    inds.append(self.nextInds[i])
        self.Inds=inds

    def getProportion(self):
        if self.isMin:
            self.Inds.sort()
        else:
            self.Inds.sort(reverse=True)
        num=1
        population=len(self.Inds)

        for i in range(1,population):
            if self.Inds[i].y!=self.Inds[0].y:
                break
            num+=1
        return num/population

    def evolution(self,generation=100,maxProportion=0.8):
        ratio=0
        num=0
        while num<generation and ratio<maxProportion:
            self.aberrance(self.K)
            self.exchange(self.CR)
            self.select()
            ratio=self.getProportion()
            print('进化代数{0}，最优值{1}，最优点{2}，最优值占比{3}'.format(num,self.Inds[0].y,self.Inds[0].x,ratio))
            num+=1
        return self.Inds[0]

if __name__=='__main__':
    def func(X):
        x=X[0]
        y=X[1]
        # return -(100*(x1**2-x2)**2+(1-x1)**2)
        return 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)-10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)-1/3*np.exp(-(x+1)**2-y**2)
    min=np.array([-3,-3])
    max=np.array([3,3])
    test=DE(min,max,100,0.5,0.5,func,True)
    ind=test.evolution(generation=100)

    mlab.points3d(ind.x[0],ind.x[1],ind.y,scale_factor=0.1)

    x,y=np.mgrid[-3:3:100j,-3:3:100j]
    s=np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a=[x[i,j],y[i,j]]
            s[i,j]=func(a)
    surf=mlab.surf(x,y,s)
    mlab.outline()
    mlab.axes(xlabel='x',ylabel='y',zlabel='z')
    mlab.colorbar()
    mlab.show()


