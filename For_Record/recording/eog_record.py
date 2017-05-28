# -*- coding: utf-8 -*-
import socket
import string
import numpy as np
import threading
import time
from datetime import datetime
from JinsSocket import JinsSocket
import pandas as pd
import math
import matplotlib.pyplot as plt
import os

import pygame
from pygame.locals import *
import sys


pygame.init()
screen = pygame.display.set_mode((300,150))
pygame.display.set_caption('MEME RECORD')
pygame.display.flip()


jins_client = JinsSocket(isUDP=True)
jins_client.setConnection()
jins_client.start()


#J!NS Data
Count = np.zeros(0)
AccX = np.zeros(0)
AccY = np.zeros(0)
AccZ = np.zeros(0)
GyroX = np.zeros(0)
GyroY = np.zeros(0)
GyroZ = np.zeros(0)
EogH = np.zeros(0)
EogV = np.zeros(0)

# record_mode = np.zeros(0)

Data = np.zeros(0)


while True:
	JinsData = jins_client.getJinsRawData()
	time.sleep(0.02) #50 Hz   

	for event in pygame.event.get():
		if event.type == QUIT: sys.exit()
		if event.type == KEYDOWN:
			if event.key == K_RIGHT: #record start
				record_mode = 1.0
			if event.key == K_LEFT: #record finish
				record_mode = 0.0
			# if event.key == K_ESCAPE:
				# DATA = pd.DataFrame()
				# DATA = Record_Data
				# DATA = pd.DataFrame(columns = ['NUM','DATE','ACC_X','ACC_Y','ACC_Z','EOG_L1','EOG_R1','EOG_L2','EOG_R2','EOG_H1','EOG_H2','EOG_V1','EOG_V2'])
				# DATA['DATE','DATE2','DATE'] = Record_Data
				# DATA.to_csv("test2.csv")

	try:
		Count = JinsData[0]
		AccX = JinsData[1]
		AccY = JinsData[2]
		AccZ = JinsData[3]
		GyroX = JinsData[4]
		GyroY = JinsData[5]
		GyroZ = JinsData[6]
		EogH = JinsData[7]
		EogV = JinsData[8]

		#meme data
		SampleCount = np.mean(Count)
		Acc_x = np.mean(AccX)
		Acc_y = np.mean(AccY)
		Acc_z = np.mean(AccZ)
		Gyro_x = np.mean(GyroX)
		Gyro_y = np.mean(GyroY)
		Gyro_z = np.mean(GyroZ)
		Vh = np.mean(EogH)
		Vv = np.mean(EogV)
		Vh_std = np.std(EogH)
		Vv_std = np.std(EogV)
		Vh_diff = np.sum(np.diff(EogH))
		Vv_diff = np.sum(np.diff(EogV))
		Vh_sum = np.sum(np.abs(np.diff(EogH)))
		Vv_sum = np.sum(np.abs(np.diff(EogV)))

		Gyro_x_diff = np.sum(np.diff(GyroX))
		Gyro_y_diff = np.sum(np.diff(GyroY))
		Gyro_z_diff = np.sum(np.diff(GyroZ))


		Data = [Vh,Vv,Vh_std,Vv_std,Vh_diff,Vv_diff,Acc_x,Acc_y,Acc_z,Gyro_x,Gyro_y,Gyro_z,record_mode]
		print Data

	except:
		None











