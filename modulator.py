#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pylab as pl

class Modulator:
    '调制器 将信息合成一个信号'

    def __init__(self, sigs=[],sampling_rate=8000):
        self.sigs=sigs
        self.sampling_rate=sampling_rate


    def load(self,sig):
        self.sigs.append(np.array(sig))
        # print("loading signal ")
        # print(np.array(sig))

    def loadImpl(self):
        res=np.zeros(len(self.sigs[0]))
        for i in range(len(self.sigs)):
            # print("dddd")
            # print(np.array(self.sigs[i]))
            res = res+ np.array(self.sigs[i])
        self.sigSum=res

    def display(self,size):
        self.loadImpl()
        t = np.arange(0, 1.0, 1.0 / self.sampling_rate)


        figNum=len(self.sigs)*100+111
        pl.figure(figsize=(8, 8))
        for i in range(len(self.sigs)):
            pl.subplot(figNum)
            pl.plot(t[:size], self.sigs[i][:size])
            pl.xlabel("time(sec)")
            pl.title("sig"+str(i+1))
            figNum+=1


        pl.subplot(figNum)
        pl.plot(t[:size], self.sigSum[:size])
        pl.xlabel("time(sec)")
        pl.title("orig")
        pl.subplots_adjust(hspace=0.4)

        pl.show()  # 找 在fft_size个取样中形成整数个周期


