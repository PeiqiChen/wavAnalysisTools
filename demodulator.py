#!/usr/bin/python
# -*- coding: UTF-8 -*-

from scipy.io import wavfile
import numpy as np
import pylab as pl
from scipy import signal


class Demodulator:
    '解调器 提取信号中的信息'
    sampling_rate = None
    fft_size =None
    path = ""
    raw_sig=None
    freqs=None

    def __init__(self, path, sampling_rate,fft_size):
        self.path = path
        self.sampling_rate = sampling_rate
        self.fft_size = fft_size

    def get_time_track(self,ifDoubleTrack):  # 识别双声道 单声道的data就是一个Array<int>
        if ifDoubleTrack == 0:
            samplimg_freq, audio = wavfile.read(self.path)
            print(len(audio))
            # pl.plot(np.arange(audio.shape[0])[:self.fft_size], audio[:self.fft_size])
            # pl.show() # 找 在fft_size个取样中形成整数个周期
            self.raw_sig=audio
        else:  # split wave
            samplerate, data = wavfile.read(self.path)
            left = []
            right = []
            for item in data:
                left.append(item[0])
                right.append(item[1])
            wavfile.write('left.wav', samplerate[:self.fft_size], np.array(left))
            wavfile.write('right.wav', samplerate, np.array(right))
            self.raw_sig={
                left:left,
                right:right
            }

    '''第二个参数0只分析不画图 1分析且画图'''
    def to_frequency(self,tm,enablePlt):
        t = np.arange(0, 1.0, 1.0 / self.sampling_rate)  # np.arange产生1秒钟的取样时间，t中的每个数值直接表示取样点的时间，因此其间隔为取样周期1/sampline_rate
        # x = np.sin(2 * np.pi * 156.25 * t) + 2 * np.sin(2 * np.pi * 234.375 * t)
        x = tm
        xs = x[:self.fft_size]
        xf = np.fft.rfft(xs) / self.fft_size  # rfft函数的返回值是N/2+1个复数，分别表示从0(Hz)到sampling_rate/2(Hz)的N/2+1点频率的成分
        freqs = np.linspace(0, self.sampling_rate / 2, self.fft_size / 2 + 1)
        xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
        freqs=range(len(xfp))
        self.freqs=xfp
        if enablePlt==1:
            pl.figure(figsize=(8, 4))
            pl.subplot(211)
            pl.plot(t[:self.fft_size], xs)
            pl.xlabel(u"time(sec)")
            pl.title(self.path)
            pl.subplot(212)
            pl.plot(freqs, xfp)
            pl.xlabel(u"frequencey(Hz)")
            pl.subplots_adjust(hspace=0.4)
            pl.show()


    '''Wn的计算说明
        这里假设采样频率为1000hz,信号本身最大的频率为500hz，
        要滤除400hz以上频率成分，即截至频率为400hz,
        则wn=2*400/1000=0.8。Wn=0.8'''
    def find3DB(self):
        peaks=[] # 波峰
        for idx in range(1,len(self.freqs)-1):
            # print(idx,self.freqs[idx - 1], self.freqs[idx], self.freqs[idx + 1])

            if self.freqs[idx - 1] <= self.freqs[idx] and  self.freqs[idx]>=self.freqs[idx + 1]:
                peaks.append((idx, self.freqs[idx]))
        print(peaks)
        bands=[] # 每个波峰对应一个带宽段
        for i in range(len(peaks)):
            val=pow(pow(peaks[i][1],2)/2,0.5) # value 3db
            low= 0 if i==0 else bands[i-1][1]
            # high=len(peaks)-2 if i==len(peaks)-2 else bands[i+1][0]
            mid=peaks[i][0]
            # print(val,low,mid,len(self.freqs[low:mid]),len(self.freqs[mid:]))

            bands.append((np.argmin(abs(self.freqs[low:mid]-val))+low,
                np.argmin(abs(self.freqs[mid:]-val))+mid,val))
        print(bands)
        return bands


    def findTroughs(self):
        troughs=[] # 波谷
        for idx in range(1,len(self.freqs)-1):
            # print(idx,self.freqs[idx - 1], self.freqs[idx], self.freqs[idx + 1])

            if self.freqs[idx - 1] >= self.freqs[idx] and  self.freqs[idx]<=self.freqs[idx + 1]:
                troughs.append((idx, self.freqs[idx]))
        print(troughs)
        return troughs


    ## 低通滤波
    def lowpass(self,gradiant,flt):
        Wn = 2 * flt / self.sampling_rate
        b, a = signal.butter(gradiant,Wn, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
        filtedData = signal.filtfilt(b, a, self.raw_sig)  # data为要过滤的信号
        return filtedData
    ## 高通滤波
    def highpass(self,gradiant,flt):
        Wn = 2 * flt / self.sampling_rate
        b, a = signal.butter(gradiant, Wn, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
        filtedData = signal.filtfilt(b, a, self.raw_sig)  # data为要过滤的信号
        return filtedData
    ## 带通滤波
    def bandpass(self, gradiant, frm,to):
        frmWn = 2 * frm / self.sampling_rate
        toWn = 2 * to / self.sampling_rate
        b, a = signal.butter(gradiant, [frmWn,toWn], 'bandpass')  # 配置滤波器 8 表示滤波器的阶数
        filtedData = signal.filtfilt(b, a, self.raw_sig)  # data为要过滤的信号
        return filtedData
    ## 带阻滤波
    def bandstop(self, gradiant, frm,to):
        frmWn = 2 * frm / self.sampling_rate
        toWn = 2 * to / self.sampling_rate
        b, a = signal.butter(gradiant, [frmWn,toWn], 'bandstop')  # 配置滤波器 8 表示滤波器的阶数
        filtedData = signal.filtfilt(b, a, self.raw_sig)  # data为要过滤的信号
        return filtedData


