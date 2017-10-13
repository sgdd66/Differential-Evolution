#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""自适应多目标差分进化算法,目标必须在两个以上"""

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

        self.K=np.random.uniform(0,1.2)
        self.dominateSet=[]
        self.dominatedNum=0
        # 非支配序
        self.rank=0
        #拥挤距离
        self.crowdingDistance=0


    def getValue(self,environment):
        self.y=environment(self.x)


    #比较函数用于用于确定优劣，调用sort函数之后按照从优到劣的顺序排序
    #原有的sort函数是按照从小到大的顺序排序的
    #所以pareto序号越大，拥挤距离越小的一方越大
    def __cmp__(self, other):
        if (self.y > other.y):
            return 1
        elif (self.y < other.y):
            return -1
        elif (self.y == other.y):
            return 0

    def __gt__(self, other):
        if self.rank>other.rank:
            return True
        elif self.rank==other.rank:
            if self.crowdingDistance<other.crowdingDistance:
                return True
        return False


    def __lt__(self, other):
        if self.rank < other.rank:
            return True
        elif self.rank == other.rank:
            if self.crowdingDistance > other.crowdingDistance:
                return True
        return False

    def __eq__(self, other):
        if self is other:
            return True
        else:
            return False

    def __deepcopy__(self, memodict={}):
        new=Ind(location=self.x)
        new.crowdingDistance=self.crowdingDistance
        new.dominatedNum=self.dominatedNum
        new.y=self.y
        new.K=self.K
        new.rank=self.rank
        new.dominateSet=self.dominateSet
        return new

    def betterThan(self,other,isMin):
        dimension=self.y.shape[0]
        if isMin:
            for i in range(dimension):
                if self.y[i]>other.y[i]:
                    return False
            if (self.y==other.y).all():
                return False
            return True
        else:
            for i in range(dimension):
                if self.y[i]<other.y[i]:
                    return False
            if (self.y == other.y).all():
                return False
            return True





class DE(object):
    def __init__(self,min,max,population,CR,environment,isMin=True):
        """min,max是一维数组，用以确定搜索空间的范围
        population是种群数量
        environment是函数指针，用以表示环境，检验个体的性能
        isMin用以确定是不是求最小值，默认为TRUE。若为FALSE表示求取最大值"""
        if(min.shape[0]>1):
            samples=DOE.LatinHypercube(dimension=min.shape[0],num=population,min=min,max=max)
            self.Inds=[]
            for i in range(population):
                ind=Ind(location=samples.realSamples[i,:])
                self.Inds.append(ind)
        else:
            self.Inds = []
            for i in range(population):
                ind = Ind(min,max)
                self.Inds.append(ind)
        self.environment=environment
        self.isMin=isMin
        self.CR=CR
        self.min=min
        self.max=max

    def aberrance(self):
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
            K=(self.Inds[i].K+self.Inds[r1].K+self.Inds[r2].K+self.Inds[r3].K)/4
            K=K*np.exp(-self.gen/3*np.abs(np.random.normal()))
            x=self.Inds[r1].x+K*(self.Inds[r2].x-self.Inds[r3].x)
            ind=Ind(location=x)
            ind.K=K
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
            ind.K=(inds1[i].K+inds2[i].K)/2

            # #多项式变异
            # eta=20
            # Pm=1/ind.x.shape[0]
            # for j in range(dimension):
            #     randk=np.random.uniform()
            #     if randk<0.5:
            #         theta=(2*randk)**(1/(eta+1))-1
            #     else:
            #         theta=1-(2-2*randk)**(1/(eta+1))
            #     randm=np.random.uniform()
            #     if randm<Pm:
            #         ind.x[j]=ind.x[j]+theta*(self.max[j]-self.min[j])

            for j in range(dimension):
                if(ind.x[j]<self.min[j]):
                    ind.x[j]=self.min[j]
                if ind.x[j]>self.max[j]:
                    ind.x[j]=self.max[j]

            inds.append(ind)
        self.nextInds=inds

    def select_tournament(self):
        """联赛选择机制，首先计算个体函数值，借此划分Pareto序号和拥挤度距离
        最后采用联赛机制，选择下一代个体"""

        population = len(self.nextInds)
        inds=self.Inds+self.nextInds

        #注意这里的0.1是一个很重要的参数，代表每次参与联赛选择比赛的人数
        tournamentNum=int(0.1*len(inds))
        Inds=[]
        for i in range(population):
            rands=np.random.randint(0,population*2,tournamentNum)
            tour=[]
            for rand in rands:
                tour.append(inds[rand])
            for i in range(tournamentNum-1,0,-1):
                if tour[i]<tour[i-1]:
                    ind = tour[i]
                    tour[i] = tour[i - 1]
                    tour[i - 1] = ind
            Inds.append(tour[0].__deepcopy__())

        self.Inds=Inds

    def select_compete(self):
        """父代个体与子代个体，列表对应位置根据Pareto序号和拥挤距离竞争"""
        population = len(self.nextInds)
        inds=[]
        for i in range(population):
            if self.Inds[i]<self.nextInds[i]:
                inds.append(self.Inds[i])
            else:
                inds.append(self.nextInds[i])
        self.Inds=inds

    def evolution(self,maxGen=100):
        self.maxGen=maxGen
        self.gen=0
        while self.gen<maxGen:
            self.aberrance()
            self.exchange(self.CR)
            population = len(self.nextInds)
            for i in range(population):
                self.Inds[i].getValue(self.environment)
                self.nextInds[i].getValue(self.environment)
            self.pareto()
            self.crowdingDistance()
            self.select_compete()

            self.Inds.sort()
            for i in range(len(self.Inds)):
                if self.Inds[i].rank!=0:
                    num=i
                    break



            print('进化代数{0}，最优值{1}，最优点{2}，Pareto Ratio:{3}'.format(self.gen,self.Inds[0].y,self.Inds[0].x,num/len(self.Inds)))
            self.gen+=1
        return self.Inds[0]

    def pareto(self):
        """快速非支配排序算法，最终得到的成员变量paretos是一个列表
        其中每一元素是一个pareto集合，按从头到尾的顺序是第一支配集到第n支配集
        每个pareto集合是ind（个体）"""

        Inds=self.Inds+self.nextInds
        population = len(Inds)
        for ind in Inds:
            ind.dominateSet=[]
            ind.dominatedNum=0
        for i in range(population-1):
            for j in range(i+1,population):
                if Inds[i].betterThan(Inds[j],self.isMin):
                    Inds[i].dominateSet.append(Inds[j])
                    Inds[j].dominatedNum+=1
                if Inds[j].betterThan(Inds[i],self.isMin):
                    Inds[j].dominateSet.append(Inds[i])
                    Inds[i].dominatedNum += 1


        paretos=[]
        rank=0
        while(len(Inds)!=0):
            pareto=[]
            for i in range(len(Inds)):
                if Inds[i].dominatedNum==0:
                    pareto.append(Inds[i])
                    for ind in Inds[i].dominateSet:
                        ind.dominatedNum-=1
            for ind in pareto:
                Inds.remove(ind)
                ind.rank=rank
            paretos.append(pareto)
            rank+=1
        self.paretos=paretos

    def bubbleSort(self,inds,objectIndex):
        #冒泡排序，从小到大升序
        population=len(inds)
        for i in range(population-1):
            for j in range(population-i-1):
                if inds[j].y[objectIndex]>inds[j+1].y[objectIndex]:
                    ind=inds[j]
                    inds[j]=inds[j+1]
                    inds[j+1]=ind

    def crowdingDistance(self):
        inds=self.Inds+self.nextInds
        for ind in inds:
            ind.crowdingDistance=0
        objectNum=inds[0].y.shape[0]
        for i in range(objectNum):
            self.bubbleSort(inds,i)
            for j in range(len(inds)):
                if j==0 or j==len(inds)-1:
                    inds[j].crowdingDistance+=1000000
                else:
                    inds[j].crowdingDistance+=(inds[j+1].y[i]-inds[j-1].y[i])/(inds[len(inds)-1].y[i]-inds[0].y[i])


if __name__=='__main__':
    def func(X):
        x=X[0]
        y=X[1]
        # return -(100*(x1**2-x2)**2+(1-x1)**2)
        return 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)-10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)-1/3*np.exp(-(x+1)**2-y**2)
    def SCH(x):
        y=np.zeros(2)
        y[0]=x[0]**2
        y[1]=(x[0]-2)**2
        return y
    # min = np.array([-1000])
    # max = np.array([1000])

    def ZDT1(x):
        y=np.zeros(2)
        y[0]=x[0]
        n=x.shape[0]
        g=1+9/(n-1)*np.sum(x[1:-1])
        y[1]=g*(1-np.sqrt(x[0]/g))
        return y
    min=np.zeros(30)
    max=np.zeros(30)+1


    test=DE(min,max,100,0.5,ZDT1,True)
    test.evolution(maxGen=100)

    # mlab.points3d(ind.x[0],ind.x[1],ind.y,scale_factor=0.1)
    #
    # x,y=np.mgrid[-3:3:100j,-3:3:100j]
    # s=np.zeros_like(x)
    # for i in range(x.shape[0]):
    #     for j in range(x.shape[1]):
    #         a=[x[i,j],y[i,j]]
    #         s[i,j]=func(a)
    # surf=mlab.surf(x,y,s)
    # mlab.outline()
    # mlab.axes(xlabel='x',ylabel='y',zlabel='z')
    # mlab.colorbar()
    # mlab.show()