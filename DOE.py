#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""实验设计方法，主要包含伪蒙特卡洛采样方法
准蒙特卡洛采样方法，拉丁超立方设计方法，
正交表采样方法，均匀采样方法"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from scipy.special import comb, perm

class PseudoMonteCarlo(object):
    """伪蒙特卡洛采样方法，考虑到均匀性，应用先分层再随机抽样的方法,采样空间在[0,1]**n之间"""
    def __init__(self,bins,num,min=None,max=None):
        """bins是n位向量，n是采样的维度，每一位的数据代表这一维分割的“箱子”数目；
        num代表这一个箱子中采样点数目，则采样点总数是bins[0]*bin[1]*...*bin[n-1]*num"""
        dimension=bins.shape[0]
        lengths=np.zeros(dimension)
        binNum=1
        for i in range(0,dimension):
            lengths[i]=1/bin[i]
            binNum=binNum*bin[i]
        binLocation=np.zeros((binNum,dimension))
        for i in range(0,binNum):
            index=self.transfer(bins,i)
            for j in range(0,dimension):
                binLocation[i,j]=index[j]*lengths[j]
        samples=np.zeros((binNum*num,dimension))
        for i in range(0,binNum):
            for j in range(0,num):
                random=np.random.uniform(0,1,dimension)
                for k in range(0,dimension):
                    samples[i*num+j,k]=binLocation[i,k]+lengths[k]*random[k]
        self.samples=samples
        if(min==None and max==None):
            self.realSamples=samples
        else:
            realSamples=np.zeros((samples.shape))
            for i in range(0,dimension):
                realSamples[:,i]=samples[:,i]*(max[i]-min[i])+min[i]
            self.realSamples=realSamples

    def transfer(self,weights,num):
        """根据weights给出的每一位的权重，将一个10进制数num转换为一个特殊进制的数
        如果num超过了weights所能表示的最大的数，则只返回可以表示的相应位置的数"""
        dimension=weights.shape[0]
        answer=np.zeros(dimension)
        remainder=num
        for i in range(0,dimension):
            answer[i]=remainder%weights[i]
            remainder=remainder//weights[i]
        return answer

class LatinHypercube(object):
    """拉丁超立方采样方法"""
    def __init__(self,dimension,num,min=None,max=None):
        """dimension是采样空间的维度，num是样本的个数，样本空间是[0,1]**dimension"""
        location=np.zeros((num,dimension))
        lists=np.arange(0,num,1)
        per=np.zeros((dimension*4,num))
        for i in range(0,per.shape[0]):
            per[i,:]=np.random.uniform(0,10,num)
        per=per.argsort(1)
        perNum=per.shape[0]
        isOK=True
        while(isOK):
            for i in range(0,dimension):
                rand=np.random.randint(0,perNum)
                location[:,i]=per[rand,:]
            try:
                R=np.cov(location.T)
                D=np.diag(np.diag(R)**-0.5)
                R=np.dot(D,np.dot(R,D))

                D=np.linalg.cholesky(R)
                G=np.dot(np.linalg.inv(D),location.T)
                isOK=False
            except np.linalg.linalg.LinAlgError as msg:
                print(msg)
                isOK=True
        list1=G.argsort(-1).T
        length=1/num
        sample=np.zeros((num,dimension))
        for i in range(0,num):
            random=np.random.uniform(0,1,dimension)
            sample[i,:]=random*length+list1[i,:]*length
        self.samples=sample
        if(min is None and max is None):
            self.realSamples=sample
        else:
            realSamples=np.zeros((sample.shape))
            for i in range(0,dimension):
                realSamples[:,i]=sample[:,i]*(max[i]-min[i])+min[i]
            self.realSamples=realSamples

if __name__=="__main__":
    # #伪蒙特卡洛采样方法测试
    # bin=np.array([10,10])
    # test=PseudoMonteCarlo(bin,1)
    # plt.scatter(test.samples[:,0],test.samples[:,1])
    # plt.show()
    #拉丁超立方采样方法测试
    test=LatinHypercube(dimension=2,num=10)
    list1=np.linspace(0,1,11)
    plt.scatter(test.samples[:,0],test.samples[:,1])
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xticks(list1)
    plt.yticks(list1)
    plt.grid(True)
    plt.show()