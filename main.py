#!/usr/bin/python
# -*- coding: UTF-8 -*-

from scipy.io import wavfile
import numpy as np
import matplotlib.pylab as plt
import pylab as pl

from demodulator import Demodulator
from modulator import Modulator

start = 0
end = 56 #
ifDoubleTrack=0 # 是否按照双声道文件解析
sampling_rate = 8000 # 采样频率
fft_size = (end-start)*100 # 分析多长的时域信号
path="singleTrack.wav"
flt=700



if __name__ == "__main__":
    dm=Demodulator(path, sampling_rate,fft_size)
    dm.get_time_track(ifDoubleTrack)
    dm.to_frequency(dm.raw_sig,1) # 0不画图 1画图

    lp=dm.lowpass(8,flt)
    hp=dm.highpass(8,flt)
    dm.to_frequency(lp,0)


    m=Modulator()

    # # 根据3DB带宽滤波 但并没有成功
    # bands = dm.find3DB()
    # for band in bands:
    #     bp=dm.bandpass(8,band[0],band[1])
    #     print(len(bp))
    #     m.load(bp)


    # 根据高通低通滤波
    trs = dm.findTroughs()
    for tr in trs:
        lp = dm.lowpass(8, tr[0])
        hp = dm.highpass(8, tr[0])
        m.load(lp)
        m.load(hp)

    res=m.display(200)




