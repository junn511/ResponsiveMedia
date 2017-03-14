# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
from matplotlib.pylab import plt
import random
import math
import string
import matplotlib.animation as animation
import threading
import socket

class JinsSocket(threading.Thread):
    def __init__(self, isUDP = False):
        threading.Thread.__init__(self)
        
        # set your default ip&port  
        # self.IP = socket.gethostbyname(socket.gethostname())
        self.IP = "127.0.0.1"
        self.Port = 12562
        self.isUDP = isUDP
        
        """ Define the length of different mode array """
        self.FULL_COUNT = 8
        
        """ Define Array """
        
        self.Count = np.zeros(0)
        self.EogL = np.zeros(0)
        self.EogR = np.zeros(0)
        self.EogH = np.zeros(0)
        self.EogV = np.zeros(0)  
        self.GyroX = np.zeros(0)
        self.GyroY = np.zeros(0)  
        self.GyroZ = np.zeros(0)

        self.JinsRawData = np.zeros(0)
        
        self.w_size = 10
        
    def setIP_Port(self, IP, Port):
        self.IP = IP
        self.Port = Port
        
    def setConnection(self):

#==============================================================================
#         [UDP]setting the socket comunication
#==============================================================================
        if self.isUDP:
            try:
                self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.serverSocket.bind((self.IP, self.Port))
            except:   
                self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.serverSocket.bind((self.IP, self.Port))
#==============================================================================
#         [TCP]setting the socket comunication
#==============================================================================
        else:
            try:
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client.connect((self.IP, self.Port))
            except:
                self.client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client.connect((self.IP, self.Port))
   
    def run(self):
        while True:
            if self.isUDP:
                data, address = self.serverSocket.recvfrom(2048)
#                data = bytes(data, 'UTF-8')
            else:
                data = self.client.recv(2048)
#                data = bytes(self.client.recv(2048), 'UTF-8')
            converted = data.decode("utf-8")
#            print(converted)
            lines = converted.split("\r\n")
            for line in lines:
#                values = string.split(line, ",")
                values = list(map(int, line.split(",")))
                # print(values)
                if len(values) == 9:
                    self.addFULL(values)
                    self.checkSIZE()
                else:
                    break
        
    
    def Disconnect(self):
        self.client.close()

    """ Full Mode """
    def addFULL(self, full_values):
        self.Count = np.append(self.Count,full_values[0])
        self.GyroX = np.append(self.GyroX,full_values[4])
        self.GyroY = np.append(self.GyroY,full_values[5])
        self.GyroZ = np.append(self.GyroZ,full_values[6])
        self.EogL = np.append(self.EogL,full_values[7])
        self.EogR = np.append(self.EogR,full_values[8])
        self.EogH = np.append(self.EogH,full_values[7]-full_values[8])
        self.EogV = np.append(self.EogV,-(full_values[7]+full_values[8])/2)
        
        self.JinsRawData = [self.Count,self.GyroX,self.GyroY,self.GyroZ,self.EogH,self.EogV]

    def checkSIZE(self):
        if len(self.EogL) > self.w_size:
            self.Count = self.Count[-self.w_size:]
            self.GyroX = self.GyroX[-self.w_size:]
            self.GyroY = self.GyroY[-self.w_size:]
            self.GyroZ = self.GyroZ[-self.w_size:]
            self.EogL = self.EogL[-self.w_size:]
            self.EogR = self.EogR[-self.w_size:]
            self.EogH = self.EogH[-self.w_size:]
            self.EogV = self.EogV[-self.w_size:]
        else:
            self.Buffer = np.array([0])

    def getJinsRawData(self):
        return self.JinsRawData

    

#==============================================================================
# Example CODE
#==============================================================================
# jins_client = JinsSocket(isUDP=True)
# jins_client.setConnection()
# jins_client.start()
# 
#==============================================================================











