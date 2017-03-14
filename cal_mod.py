# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import peakutils

from os import listdir, remove
from datetime import datetime, timedelta

import math
import scipy.signal as signal
import sys


def getMaxCorr(BASE, h):
    """Calculate correlation of all candidate of orbit position 
    and get the maximum value of correlation & its index"""
    
    results = np.array([])
    for base in BASE:
        results = np.append(results,np.corrcoef(base, h)[0,1])
    
    index = results.argmax()
    value = results[index]    
    
    return index, value
    


def fftPeakDetect(h,N):
    """Detect the peak point index of frequency domain"""
    
    H= np.abs(np.fft.fft(h[-N:])[0:N])
    #H=HAN*H
    
    indexes = peakutils.indexes(H, thres=0.5, min_dist=30)
    if len(indexes)==0:
        M_AVG=0
    else:
        M_AVG=H[indexes[0]]/np.average(H)

    return H, indexes, M_AVG
    
###################################
"""DATA LOADING"""

def getJinsDATA(my_file, file_path):
    print 'open JINS file:'+my_file
    this_f = open(my_file,'r')
    lines = this_f.readlines()
    new_f = open(file_path+"tmp.csv", 'w')
    for i in range(5,len(lines)):
        new_f.write(lines[i])
    this_f.close()
    new_f.close()    
    
    """JINS DATA"""
    df_jins = pd.DataFrame.from_csv(file_path+"tmp.csv")
    remove(file_path+"tmp.csv")
    
    dates_s = df_jins['DATE']
    dates = []
    for date in dates_s:
        try:
            if type(date)==str:
                tmp = datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
            else:
                print date
        except ValueError as e:
            tmp = datetime.strptime(date, '%Y/%m/%d %H:%M:%S.%f')
        dates.append(tmp)
        
    df_jins['DATE'] = dates
    
    return df_jins

def getOrbitDATA(file_path, deltaHwithJins):
    print 'open Orbit file:'+file_path
    df_orbit = pd.DataFrame.from_csv(file_path)
    dates_s = df_orbit['time']
    dates = []
    for date in dates_s:
        tmp = datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
        micros = int(np.round(tmp.microsecond/10000.0))
        if micros == 100:
            micros = 99
        tmp = tmp.replace(microsecond=micros*10000)
        tmp = tmp + timedelta(hours=deltaHwithJins)
        dates.append(tmp)
        
    df_orbit['time'] = dates
    
    return df_orbit
    
###################################
"""ORBIT DATA RECOVERY"""

def makingFalseOrbit(periods, counter_clks, starting_thetas, trans_speed, N):
    candidates_num = len(periods)*len(counter_clks)*len(starting_thetas)
    
    candidate_orbitX = np.zeros((candidates_num, N))
    candidate_orbitY = np.zeros((candidates_num, N))
    candidate_theta = np.zeros((candidates_num, N))
    
    index = 0
    for period in periods:
        for counter_clk in counter_clks:
            for start_theta in starting_thetas:
                candidate_orbitX[index,:], candidate_orbitY[index,:], candidate_theta[index,:] = makingOrbitXY_N(start_theta,1,period,counter_clk,trans_speed,N)
                index+=1
                
    return candidate_orbitX, candidate_orbitY, candidate_theta
    
def makingOrbitXY_N(start_theta, r, period, counter_clk, transmission_speed, N):
    cur_theta = start_theta
    d_theta = 2*np.pi/(10*period/transmission_speed)
    
    theta = np.zeros(N)
    o_x = np.zeros(N)
    o_y = np.zeros(N)
    for i in range(N):
        theta[i] = cur_theta
        o_x[i] = r*np.cos(cur_theta)
        o_y[i] = r*np.sin(cur_theta)
        
        cur_theta += counter_clk*d_theta
        
    return o_x, o_y, theta
    
    
def makingOrbitXY(df_jins, df_orbit, transmission_speed):
    o_time = df_orbit['time']
    j_time = df_jins['DATE']
    
    cur_mode = -1
    flw_mode = []
    
#    o_x = np.array([])
#    o_y = np.array([])
    
    length = len(j_time)
    o_x = np.zeros(length)
    o_y = np.zeros(length)
    flw_mode = np.zeros(length)
    index = 0
    
    o_index = 0
    r = 0.0
    theta = 0.0
    d_theta = 0.0
    counter_clk = 0.0
    for cur_time in j_time:
        if o_index < len(o_time) and(cur_time == o_time[o_index] or cur_time > o_time[o_index]):
            date,r,theta,period,counter_clk,cur_mode = df_orbit.loc[o_index] 
            o_index += 1
            d_theta = 2*np.pi/(10*period/transmission_speed)
#            d_theta = 2*np.pi/(period/10)
            if counter_clk == 0:
                counter_clk = -1
        else:
            theta += counter_clk*d_theta
        
        if cur_mode == -1:
#            o_x = np.append(o_x,-100)
#            o_y = np.append(o_y,-100)
            o_x[index] = -100
            o_y[index] = -100
        else:
#            o_x = np.append(o_x,r*np.cos(theta))
#            o_y = np.append(o_y,r*np.sin(theta))
            o_x[index] = r*np.cos(theta)
            o_y[index] = r*np.sin(theta)
            
        flw_mode[index] = cur_mode
        index += 1
        
#        flw_mode.append(cur_mode)
            
    return o_x, o_y, flw_mode
    
def makingOrbitXY_false(df_jins, df_orbit, transmission_speed):
    o_time = df_orbit['time']
    j_time = df_jins['DATE']
    
    cur_mode = -1
    flw_mode = []
    
#    o_x = np.array([])
#    o_y = np.array([])
    
    length = len(j_time)
    o_x = np.zeros(length)
    o_y = np.zeros(length)
    flw_mode = np.zeros(length)
    
    o_index = 0
    r = 0.0
    theta = 0.0
    d_theta = 0.0
    counter_clk = 0.0
    index = 0
    for cur_time in j_time:
        if o_index < len(o_time) and(cur_time == o_time[o_index] or cur_time > o_time[o_index]):
            date,r,theta,period,counter_clk,cur_mode = df_orbit.loc[o_index] 
            o_index += 1
            d_theta = 2*np.pi/(10*period/transmission_speed)
#            d_theta = 2*np.pi/(period/10)
            if counter_clk == 0:
                counter_clk = -1
        else:
            theta += counter_clk*d_theta
            
#        o_x = np.append(o_x,r*np.cos(theta))
#        o_y = np.append(o_y,r*np.sin(theta))
#        flw_mode.append(cur_mode)
        
        o_x[index] = r*np.cos(theta)
        o_y[index] = r*np.sin(theta)
        flw_mode[index] = cur_mode
        index += 1
            
    return o_x, o_y, flw_mode   

def makingOrbitXY_withWholedata(df_jins, df_orbit):
    o_time = df_orbit['time']
    j_time = df_jins['DATE']
    
    cur_mode = -1
    
    length = len(j_time)
    flw_mode = np.zeros(length)
    periods = np.zeros(length)
    counter_clks = np.zeros(length)
    r_sizes = np.zeros(length)
    o_x = np.zeros(length)
    o_y = np.zeros(length)
    thetas = np.zeros(length)
    heads = np.zeros(length)
    
    sequence_ids = np.zeros(length)
    cur_seq = 0
    o_index = 0
    r = 0.0
    theta = 0.0
    d_theta = 0.0
    counter_clk = 0.0
    period = 1.0
    head = 0
    index = 0
    
    prev_time = j_time[0]
    length = len(j_time)
    for index in range(length):
        cur_time = j_time[index]
        
#         set current state change
        if o_index < len(o_time) and(cur_time >= o_time[o_index]):
            date,r,theta,period,counter_clk,cur_mode,head = df_orbit.loc[o_index] 
            o_index += 1
            cur_seq += 1
            d_theta = counter_clk*2*np.pi*10/period
            if counter_clk == 0:
                counter_clk = -1
            
        d_t = cur_time - prev_time
        d_t = d_t.seconds*1000000+d_t.microseconds
        
        if d_t == 10000:
            theta += d_theta
        else:
            # update the delta to adapt the data missings...
#            print d_t
            theta += counter_clk*2*np.pi*d_t/(period*1000.0)
        
        o_x[index] = r*np.cos(theta)
        o_y[index] = r*np.sin(theta)
        
        thetas[index] = theta
        flw_mode[index] = cur_mode
        periods[index] = period
        counter_clks[index] = counter_clk
        r_sizes[index] = r
        heads[index] = head
        sequence_ids[index] = cur_seq
        
        prev_time = cur_time
#        sys.stdout.write("\r%d/%d" % (index,length))
#        sys.stdout.flush()    
            
    return thetas, o_x, o_y, flw_mode,r_sizes, periods, counter_clks, sequence_ids, heads

#not in use
def makingOrbitXY_withWholedata2(df_jins, df_orbit):
    o_time = df_orbit['time']
    j_time = df_jins['DATE']
    
    cur_mode = -1
    
    length = len(j_time)
    flw_mode = np.ones(length)*-100
    periods = np.ones(length)*-100
    counter_clks = np.ones(length)*-100
    r_sizes = np.ones(length)*-100
    o_x = np.ones(length)*-100
    o_y = np.ones(length)*-100
    thetas = np.ones(length)*-100
    heads = np.ones(length)*-100
    
    sequence_ids = np.ones(length)*-100
    cur_seq = 0
    o_index = 0
    r = 0.0
    theta = 0.0
    d_theta = 0.0
    counter_clk = 0.0
    period = 1.0
    head = 0
    index = 0
    
    prev_time = j_time[0]
    
    for index in range(length):
        cur_time = j_time[index]
        
#         set current state change
        if cur_time >= o_time[o_index]:
            date,r,theta,period,counter_clk,cur_mode,head = df_orbit.loc[o_index] 
            
            cur_seq += 1
            d_theta = counter_clk*2*np.pi*10/period
            if counter_clk == 0:
                counter_clk = -1
                
            o_x[index] = r*np.cos(theta)
            o_y[index] = r*np.sin(theta)
            thetas[index] = theta
            flw_mode[index] = cur_mode
            periods[index] = period
            counter_clks[index] = counter_clk
            r_sizes[index] = r
            heads[index] = head
            sequence_ids[index] = cur_seq
            
            o_index += 1
            if o_index >= len(o_time):
                break
        
        elif cur_mode == 1:
            d_t = cur_time - prev_time
            d_t = d_t.seconds*1000000+d_t.microseconds
            if d_t == 10000:
                theta += d_theta
            else:
                # update the delta to adapt the data missings...
                theta += counter_clk*2*np.pi*d_t/(period*1000.0)
            
            o_x[index] = r*np.cos(theta)
            o_y[index] = r*np.sin(theta)
            thetas[index] = theta
            flw_mode[index] = cur_mode
            periods[index] = period
            counter_clks[index] = counter_clk
            r_sizes[index] = r
            heads[index] = head
            sequence_ids[index] = cur_seq
            
        
        prev_time = cur_time
#        sys.stdout.write("\r%d/%d" % (index,length))
#        sys.stdout.flush()    
            
    return thetas, o_x, o_y, flw_mode,r_sizes, periods, counter_clks, sequence_ids, heads
    
    
    
    
    
###################################
"""CORRELATION CALCULATION & CLASSIFYER"""    
def getCorr(BASE, h):
    """Just Simple Correlation (same as Orbits)"""

    results = np.corrcoef(BASE, h)[0,1]
    if math.isnan(results):
        results = 0
    return results


def getCorrResults(w_size, rec_h, rec_v, or_x, or_y):
    tmp = [len(rec_h), len(rec_v), len(or_x), len(or_y)]
#    results_h = np.array([])
#    results_v = np.array([])
    results_h = np.zeros(tmp[0])
    results_v = np.zeros(tmp[0])
    
    if not len(set(tmp))==1:
        print "Lengthes are different: rec_h, rec_v, or_x, or_y"
        print tmp
        
    else:
        for i in range(len(or_x)):
            if i >= w_size:
                results_h[i] = getCorr(rec_h[i-w_size:i],or_x[i-w_size:i])
                results_v[i] = getCorr(rec_v[i-w_size:i],or_y[i-w_size:i])
                
#            if i < w_size:
#                results_h = np.append(results_h,0)
#                results_v = np.append(results_v,0)
#            else:
#                results_h = np.append(results_h,getCorr(rec_h[i-w_size:i],or_x[i-w_size:i]))
#                results_v = np.append(results_v,getCorr(rec_v[i-w_size:i],or_y[i-w_size:i]))
        
    return results_h, results_v

    
def classifyOrbit(corr_h, corr_v, thre_corr, thre_per):   
    """Use Only H"""
    thre_count = [0,0]
    plus_count = [0,0]
    minus_count = [0,0]
    
    h = np.array(corr_h)
    v = np.array(corr_v)

    result = -1    
    
    tmp, = np.where(np.abs(h)>thre_corr)
    thre_count[0] = len(tmp)
    tmp, = np.where(h>0)
    plus_count[0] = len(tmp)
    tmp, = np.where(h<0)
    minus_count[0] = len(tmp)
    
    tmp, = np.where(np.abs(v)>thre_corr)
    thre_count[1] = len(tmp)
    tmp, = np.where(v>0)
    plus_count[1] = len(tmp)
    tmp, = np.where(v<0)
    minus_count[1] = len(tmp)
    
    if thre_count[0] > (len(corr_h)*thre_per):
        result = 1

    return thre_count, plus_count, minus_count, result
    
def classifyOrbit2(corr_h, corr_v, thre_corr, thre_per, thre_corr2, thre_per2):    
    """Use H,V both"""    
    thre_count = [0,0]
    plus_count = [0,0]
    minus_count = [0,0]
    
    h = np.array(corr_h)
    v = np.array(corr_v)

    result = -1    
    
    tmp, = np.where(np.abs(h)>thre_corr)
    thre_count[0] = len(tmp)
    tmp, = np.where(h>0)
    plus_count[0] = len(tmp)
    tmp, = np.where(h<0)
    minus_count[0] = len(tmp)
    
    tmp, = np.where(np.abs(v)>thre_corr2)
    thre_count[1] = len(tmp)
    tmp, = np.where(v>0)
    plus_count[1] = len(tmp)
    tmp, = np.where(v<0)
    minus_count[1] = len(tmp)
    
    
    if thre_count[0] > (len(corr_h)*thre_per):
        if thre_count[1] > (len(corr_v)*thre_per2):
            result = 1
        
    return thre_count, plus_count, minus_count, result
    
def getHanAvg(signal):
    han = np.hanning(len(signal))
    result = han*signal
    
    return result.mean()

def getHanAbsAvg(signal):
    han = np.hanning(len(signal))
    result = np.abs(han*signal)
    print result.mean()
    return result.mean()

def getClassResults(win_size, thre_corr, thre_per, results_h, results_v):
    """Use Only H"""
    results_classALL = np.zeros(len(results_h))
    
    for i in range( len(results_h)):
        if i < win_size:
            results_classALL[i] = -1
            continue
        
        tmp_thre, tmp_plus, tmp_minus, result = classifyOrbit(results_h[i-win_size:i],results_v[i-win_size:i],thre_corr, thre_per)
        results_classALL[i] = result
    
    return results_classALL

def getClassResults2(win_size, thre_corr, thre_per, thre_corr2, thre_per2, results_h, results_v):
    """Use H,V both"""
    results_classALL = np.zeros(len(results_h))
    
    for i in range( len(results_h)):
        if i < win_size:
#            results_classALL = np.append(results_classALL, -1)
            results_classALL[i] = -1
            continue
        
        tmp_thre, tmp_plus, tmp_minus, result = classifyOrbit2(results_h[i-win_size:i],results_v[i-win_size:i],thre_corr, thre_per, thre_corr2, thre_per2)
#        results_classALL = np.append(results_classALL, result)
        results_classALL[i] = result
    
    return results_classALL

class FalsePositiveTest:
#    
    def __init__(self):
        print "making..."
        
#%%
class OrbitClassify:
    
    def __init__(self, w_size, inner_w_size, thre_H, thre_V):
        tmp = int(w_size/inner_w_size)
        self.cal_indexes = [inner_w_size*(i+1) for i in range(tmp) if inner_w_size*(i+1)<=w_size]
        self.cal_indexes_len = len(self.cal_indexes)
        self.results_h = np.zeros(self.cal_indexes_len)
        self.results_v = np.zeros(self.cal_indexes_len)
        
        self.inner_w_size = inner_w_size
        self.w_size = w_size
        self.thre_H = thre_H
        self.thre_V = thre_V
        
        self.rec_h = np.array([])
        self.rec_v = np.array([])
        self.orbit_x = np.array([])
        self.orbit_y = np.array([])
        
        self.cur_i_End = 0
        self.cur_i_Start = 0
        
        self.error_seq = []
        
        self.fN  = 2    # Filter order
        self.Wn = 0.07 # Cutoff frequency
        self.B, self.A = signal.butter(self.fN, self.Wn, output='ba')
        
    def setLowPassfilter(self, fN, Wn):
        self.fN  = fN    # Filter order
        self.Wn = Wn # Cutoff frequency
        
        self.B, self.A = signal.butter(self.fN, self.Wn, output='ba')
        
    def LowPassFilter(self, in_sig):
        self.in_sig = in_sig
        passed = signal.filtfilt(self.B,self.A, self.in_sig)
        
        return passed
                
    def getCorr(self,BASE, h):
        """Just Simple Correlation (same as Orbits)"""
        results = np.corrcoef(BASE, h)[0,1]
        if math.isnan(results):
            results = 0
            
        return results
        
    def getMaxCorrOfWindow(self, rec_h, rec_v, orbit_x, orbit_y):
        
        for i in range(self.cal_indexes_len):
            cur_i_End = self.cal_indexes[i]
            cur_i_Start = cur_i_End - self.inner_w_size

            tmp1 = self.rec_h[cur_i_Start:cur_i_End]
            tmp2 = self.orbit_x[cur_i_Start:cur_i_End]
            self.results_h[i] = self.getCorr(tmp1, tmp2)
            tmp1 = self.rec_v[cur_i_Start:cur_i_End]
            tmp2 = self.orbit_y[cur_i_Start:cur_i_End]
            self.results_v[i] = self.getCorr(tmp1, tmp2)
            
        result_h = np.max(np.abs(self.results_h))
        result_v = np.max(np.abs(self.results_v))
        
        return result_h, result_v
        
    def getMaxCorrOfALL(self, rec_h, rec_v, orbit_x, orbit_y):
        res_h = np.zeros(len(rec_h))
        res_v = np.zeros(len(rec_h))
        
        for i in range(len(rec_h)):
            if i >= self.inner_w_size:
                res_h[i] = self.getCorr(rec_h[i-self.inner_w_size:i],orbit_x[i-self.inner_w_size:i])
                res_v[i] = self.getCorr(rec_v[i-self.inner_w_size:i],orbit_y[i-self.inner_w_size:i])

        result_h = np.max(np.abs(res_h))
        if len(np.where(res_h == result_h)[0]) ==0:
            result_h  *= -1
        result_v = np.max(np.abs(res_v))
        if len(np.where(res_v == result_v)[0]) ==0:
            result_v  *= -1
            
        return result_h, result_v
    
    def getMaxNCorrOfALL(self, rec_h, rec_v, orbit_x, orbit_y, N):
        res_h = np.zeros(len(rec_h)-self.inner_w_size)
        res_v = np.zeros(len(rec_h)-self.inner_w_size)
        
        for i in range(len(rec_h)-self.inner_w_size):
            index = i + self.inner_w_size
            res_h[i] = self.getCorr(rec_h[i:index],orbit_x[i:index])
            res_v[i] = self.getCorr(rec_v[i:index],orbit_y[i:index])

        result_h = np.max(np.abs(res_h))
        index = np.where(res_h == result_h)[0]
        if len(index)>1:
            print "more than 2 maximum val H"
            
        if len(index) == 0:
            index = np.where(res_h == -result_h)[0][0]
        else:
            index = index[0]
            
        if index-N < 0:
            index = N
        elif index+N >= self.w_size:
            index = self.w_size - N -1
        result_h = res_h[index-N:index+N].mean()
            
        result_v = np.max(np.abs(res_v))
        index = np.where(res_v == result_v)[0]
        if len(index)>1:
            print "more than 2 maximum val V"
        if len(index) == 0:
            index = np.where(res_v == -result_v)[0][0]
        else:
            index = index[0]
            
        if index-N < 0:
            index = N
        elif index+N >= self.w_size:
            index = self.w_size - N -1
        result_v = res_v[index-N:index+N].mean()
            
        return result_h, result_v
        
    def getMaxNCorrOfONE(self, gyro_z, orbit_x, orbit_y, N):
        res_h = np.zeros(len(gyro_z)-self.inner_w_size)
        res_v = np.zeros(len(gyro_z)-self.inner_w_size)
        
        for i in range(len(gyro_z)-self.inner_w_size):
            index = i + self.inner_w_size
            res_h[i] = self.getCorr(gyro_z[i:index],orbit_x[i:index])
            res_v[i] = self.getCorr(gyro_z[i:index],orbit_y[i:index])

        result_h = np.max(np.abs(res_h))
        result_v = np.max(np.abs(res_v))
        
        if result_h > result_v :
            index = np.where(res_h == result_h)[0]
            if len(index) == 0:
                index = np.where(res_h == -result_h)[0][0]
                result = res_h[index-N:index+N].mean()
            else:
                index = index[0]
                result = res_h[index-N:index+N].mean()
            
        else:
            index = np.where(res_v == result_v)[0]
            if len(index) == 0:
                index = np.where(res_v == -result_v)[0][0]
                result = res_v[index-N:index+N].mean()
            else:
                index = index[0]
                result = res_v[index-N:index+N].mean()
            
        return result
        
    
    def getNMaxCorrOfALL(self, rec_h, rec_v, orbit_x, orbit_y, N):
        res_h = np.zeros(len(rec_h)-self.inner_w_size)
        res_v = np.zeros(len(rec_h)-self.inner_w_size)
        
        for i in range(len(rec_h)-self.inner_w_size):
            index = i + self.inner_w_size
            res_h[i] = self.getCorr(rec_h[i:index],orbit_x[i:index])
            res_v[i] = self.getCorr(rec_v[i:index],orbit_y[i:index])

        result_h = np.max(np.abs(res_h))
        res_h.sort()   
        if result_h == res_h[-1]:
            result_h = res_h[-N:].mean()
        else:
            result_h = res_h[:N].mean()
            
            
        result_v = np.max(np.abs(res_v))
        res_v.sort()
        if result_v == res_v[-1]:
            result_v = res_v[-N:].mean()
        else:
            result_v = res_v[:N].mean()
            
        return result_h, result_v
    
    
    def getRESULT(self, rec_h, rec_v, orbit_x, orbit_y):
        self.rec_h = rec_h
        self.rec_v = rec_v
        self.orbit_x = orbit_x
        self.orbit_y = orbit_y
        res_h,res_v = self.getMaxCorrOfWindow(self.rec_h, self.rec_v, self.orbit_x, self.orbit_y)
#        res_h,res_v = self.getMaxCorrOfWindow(rec_h, rec_v, orbit_x, orbit_y)
        if np.abs(res_h)>=self.thre_H and np.abs(res_v)>=self.thre_V:
            return 1
        else:
            return 0
            
    def getRESULT2(self, rec_h, rec_v, orbit_x, orbit_y):
        self.rec_h = rec_h
        self.rec_v = rec_v
        self.orbit_x = orbit_x
        self.orbit_y = orbit_y
        res_h,res_v = self.getMaxCorrOfALL(self.rec_h, self.rec_v, self.orbit_x, self.orbit_y)
#        res_h,res_v = self.getMaxCorrOfALL(rec_h, rec_v, orbit_x, orbit_y)
        if np.abs(res_h)>=self.thre_H and np.abs(res_v)>=self.thre_V:
            return 1
        else:
            return 0
            
    def getRESULT3(self, partial_df):
        self.partial_df = partial_df
        self.rec_h = self.partial_df['EOG_H'].as_matrix()
        self.rec_v = self.partial_df['EOG_V'].as_matrix()
        self.orbit_x = self.partial_df['orbit_x'].as_matrix()
        self.orbit_y = self.partial_df['orbit_y'].as_matrix()
        
        res_h,res_v = self.getMaxCorrOfALL(self.rec_h, self.rec_v, self.orbit_x, self.orbit_y)
        
        head = self.partial_df['head_orbit'][0]
        if np.abs(res_h)>=self.thre_H and np.abs(res_v)>=self.thre_V:
            if head * res_h > 0:
                return 1
            else:
                self.error_seq.append(partial_df['sequence_id'][0])
#                print "HEAD:%d, Corr:%f" % (head,res_h)
#                print name
        else:
            return 0
    
    def getRESULT4(self, partial_df):
        self.partial_df = partial_df
        self.rec_h = self.partial_df['EOG_H'].as_matrix()
        self.rec_v = self.partial_df['EOG_V'].as_matrix()
        self.orbit_x = self.partial_df['orbit_x'].as_matrix()
        self.orbit_y = self.partial_df['orbit_y'].as_matrix()
        
        res_h,res_v = self.getMaxCorrOfALL(self.rec_h, self.rec_v, self.orbit_x, self.orbit_y)
        
        head = self.partial_df['head_orbit'][0]
        gyro_z = self.partial_df['GYRO_Z'].as_matrix()
        mean = np.mean(np.abs(gyro_z))
        std = np.std(gyro_z)
#        max_min = np.max(gyro_z)-np.min(gyro_z)
        
        if np.abs(res_h)>=self.thre_H and np.abs(res_v)>=self.thre_V:
            if head > 0 and mean*std > 420000:
                return 1,res_h,res_v
            elif head < 0 and mean*std < 420000:
                return 1,res_h,res_v
            else:
                self.error_seq.append([partial_df['sequence_id'][0],partial_df['name'][0], mean, std])
                return 0,res_h,res_v
                #                print "HEAD:%d, Corr:%f" % (head,res_h)
#                print name
        else:
            return 0,res_h,res_v
            
    def getRESULT5(self, partial_df, N):
        self.partial_df = partial_df
        self.rec_h = self.partial_df['EOG_H'].as_matrix()
        self.rec_v = self.partial_df['EOG_V'].as_matrix()
        self.orbit_x = self.partial_df['orbit_x'].as_matrix()
        self.orbit_y = self.partial_df['orbit_y'].as_matrix()
        
        res_h,res_v = self.getMaxNCorrOfALL(self.rec_h, self.rec_v, self.orbit_x, self.orbit_y, N)
        
        head = self.partial_df['head_orbit'][0]
        gyro_z = self.partial_df['GYRO_Z'].as_matrix()
        mean = np.mean(np.abs(gyro_z))
        std = np.std(gyro_z)
#        max_min = np.max(gyro_z)-np.min(gyro_z)
        
        if np.abs(res_h)>=self.thre_H and np.abs(res_v)>=self.thre_V:
            if head > 0 and mean*std > 420000:
                return 1,res_h,res_v
            elif head < 0 and mean*std < 420000:
                return 1,res_h,res_v
            else:
                self.error_seq.append([partial_df['sequence_id'][0],partial_df['name'][0]])
                return 0,res_h,res_v
                #                print "HEAD:%d, Corr:%f" % (head,res_h)
#                print name
        else:
            return 0,res_h,res_v
            
    def getRESULT6(self, partial_df, N):
        self.partial_df = partial_df
        self.rec_h = self.partial_df['EOG_H'].as_matrix()
        self.rec_v = self.partial_df['EOG_V'].as_matrix()
        self.rec_h = signal.filtfilt(self.B,self.A,self.rec_h)
        self.rec_v = signal.filtfilt(self.B,self.A,self.rec_v)
        
        self.orbit_x = self.partial_df['orbit_x'].as_matrix()
        self.orbit_y = self.partial_df['orbit_y'].as_matrix()
        
        res_h,res_v = self.getMaxNCorrOfALL(self.rec_h, self.rec_v, self.orbit_x, self.orbit_y, N)
        
        head = self.partial_df['head_orbit'].iloc[0]
        gyro_z = self.partial_df['GYRO_Z'].as_matrix()
        mean = np.mean(np.abs(gyro_z))
        std = np.std(gyro_z)
#        max_min = np.max(gyro_z)-np.min(gyro_z)
        result = 0
        
        if np.abs(res_h)>=self.thre_H and np.abs(res_v)>=self.thre_V and res_h * res_v >0:
            if head > 0 and mean*std > 420000:
                result = 1
            elif head < 0 and mean*std < 420000:
                result = 1
            else:
#                self.error_seq.append([partial_df['sequence_id'].iloc[0],partial_df['name'].iloc[0], mean, std])
                result = 2                
                #                print "HEAD:%d, Corr:%f" % (head,res_h)
#                print name
        return result,res_h,res_v,mean,std
            
    def getRESULT7(self, partial_df, N):
        self.partial_df = partial_df
        self.rec_h = self.partial_df['EOG_H'].as_matrix()
        self.rec_v = self.partial_df['EOG_V'].as_matrix()
        self.rec_h = signal.filtfilt(self.B,self.A,self.rec_h)
        self.rec_v = signal.filtfilt(self.B,self.A,self.rec_v)
        self.orbit_x = self.partial_df['orbit_x'].as_matrix()
        self.orbit_y = self.partial_df['orbit_y'].as_matrix()
        
        res_h,res_v = self.getMaxNCorrOfALL(self.rec_h, self.rec_v, self.orbit_x, self.orbit_y, N)
        
        head = self.partial_df['head_orbit'][0]
        gyro_z = self.partial_df['GYRO_Z'].as_matrix()
        
        gyro_result = self.getMaxNCorrOfONE(gyro_z, self.orbit_x, self.orbit_y, N)
        
        mean = np.mean(np.abs(gyro_z[self.w_size/3:-self.w_size/3]))
        std = np.std(gyro_z[self.w_size/3:-self.w_size/3])
#        max_min = np.max(gyro_z)-np.min(gyro_z)
        
        if np.abs(res_h)>=self.thre_H and np.abs(res_v)>=self.thre_V:
            if head * res_h > 0:
                return 1,res_h,res_v, mean, std, gyro_result
#            if head > 0 and mean*std > 420000:
#                return 1,res_h,res_v, mean, std
#            elif head < 0 and mean*std < 420000:
#                return 1,res_h,res_v, mean, std
            else:
                self.error_seq.append([partial_df['sequence_id'][0],partial_df['name'][0], mean, std])
                return 2,res_h,res_v, mean, std, gyro_result
                #                print "HEAD:%d, Corr:%f" % (head,res_h)
#                print name
        else:
            return 0,res_h,res_v, mean, std, gyro_result
            
    def printResult(self, partial_df):
        self.partial_df = partial_df
        self.rec_h = self.partial_df['EOG_H'].as_matrix()
        self.rec_v = self.partial_df['EOG_V'].as_matrix()
        self.orbit_x = self.partial_df['orbit_x'].as_matrix()
        self.orbit_y = self.partial_df['orbit_y'].as_matrix()
        
        gyro_z = self.partial_df['GYRO_Z'].as_matrix()
        max_min = np.max(gyro_z)-np.min(gyro_z)
        mean = np.mean(np.abs(gyro_z))
        std = np.std(gyro_z)
        
        res_h,res_v = self.getMaxCorrOfALL(self.rec_h, self.rec_v, self.orbit_x, self.orbit_y)
        
        head = self.partial_df['head_orbit'][0]
        print "HEAD:%d, CorrH:%f,CorrV:%f, max-min:%d, mean:%d, std:%f" % (head,res_h,res_v, max_min, mean, std)
        
    def getRESULTforFALSE(self, rec_h, rec_v, o_x, o_y):
        self.rec_h = rec_h
        self.rec_v = rec_v
        self.orbit_x = o_x
        self.orbit_y = o_y
        
        res_h,res_v = self.getMaxCorrOfALL(self.rec_h, self.rec_v, self.orbit_x, self.orbit_y)
        
        if np.abs(res_h)>=self.thre_H and np.abs(res_v)>=self.thre_V:
#            print "res_h,res_v: %f,%f"%(res_h,res_v)
            return 1,res_h,res_v
        else:
            return 0,res_h,res_v
            
    def getRESULTforFALSE_N(self, rec_h, rec_v, o_x, o_y, N):
        self.rec_h = rec_h
        self.rec_v = rec_v
        self.orbit_x = o_x
        self.orbit_y = o_y
        
        res_h,res_v = self.getMaxNCorrOfALL(self.rec_h, self.rec_v, self.orbit_x, self.orbit_y, N)
        
        if np.abs(res_h)>=self.thre_H and np.abs(res_v)>=self.thre_V:
#            print "res_h,res_v: %f,%f"%(res_h,res_v)
            return 1,res_h,res_v
        else:
            return 0,res_h,res_v
            
    def getRESULTforFALSE_N_low(self, rec_h, rec_v, o_x, o_y, N):
        self.rec_h = rec_h
        self.rec_v = rec_v
        self.rec_h = signal.filtfilt(self.B,self.A,self.rec_h)
        self.rec_v = signal.filtfilt(self.B,self.A,self.rec_v)
        self.orbit_x = o_x
        self.orbit_y = o_y
        
        res_h,res_v = self.getMaxNCorrOfALL(self.rec_h, self.rec_v, self.orbit_x, self.orbit_y, N)
        
        if np.abs(res_h)>=self.thre_H and np.abs(res_v)>=self.thre_V and res_h * res_v >0:
#            print "res_h,res_v: %f,%f"%(res_h,res_v)
            return 1,res_h,res_v
        else:
            return 0,res_h,res_v

#%%
###################################
"""GET RESULT RATE"""

def getTPrate(ground_truth, result, truth_val):
    """Get True/Positive rate
    by comparing the ground_truth and result(set the truth_value as list)"""
    total = 0
    tmp =0
    for i in range(len(ground_truth)):
        if ground_truth[i] in truth_val:
            if ground_truth[i] == result[i]:
                tmp += 1
            total += 1

    tmp = 100.0* tmp/total
    
    return tmp

def getFNrate(ground_truth, result, neg_val):
    """Get Falsed/Negative rate
    by comparing the ground_truth and result(set the negative_value)"""
    total = 0
    tmp =0
    for i in range(len(ground_truth)):
        if ground_truth[i] == neg_val:
            if ground_truth[i] == result[i]:
                tmp += 1
            total += 1

    tmp = 100.0* tmp/total
    
    return tmp

def getFPrate(results, neg_val):
    """Get Falsed/Positive rate from Natural data(whole flase data)
    by just the result(set the negative_value)"""
    total = len(results)
    tmp =0
    for result in results:
        if result != neg_val:
            tmp += 1

    tmp = 100.0* tmp/total
    return tmp
    
###################################    
"""OLD PART"""
def makingSinWave(winSize, sampRate, period, phase):
    """basis period: 1sec"""
    d = (2*np.pi/sampRate)*period
    x = [phase-(winSize-i+1)*d for i in range(winSize)]
    y = np.sin(x)
    return y

def makingCosWave(winSize, sampRate, period, phase):
    """basis period: 1sec"""
    d = (2*np.pi/sampRate*period)
    x = [phase-(winSize-i+1)*d for i in range(winSize)]
    y = np.cos(x)
    return y

def makingCorrSample(winSize, sampRate, period, cos_sin):
    """Generating all candidate of orbit position 
    to calculate correlation"""
    BASE = np.array([[]])
    d = (2*np.pi/(sampRate*period))
    
    if cos_sin=="cos":
        for i in range(int(sampRate*period)):
            if i==0:
                BASE = np.array([makingCosWave(winSize,sampRate,period,d*i)])
            else:
                BASE = np.append(BASE,[makingCosWave(winSize,sampRate,period,d*i)],axis=0)
    elif cos_sin=="sin":
        for i in range(int(sampRate*period)):
            if i==0:
                BASE = np.array([makingCosWave(winSize,sampRate,period,d*i)])
            else:
                BASE = np.append(BASE,[makingCosWave(winSize,sampRate,period,d*i)],axis=0)
    else:
        print "Type correctly... cos or sin"
    
    return BASE    
