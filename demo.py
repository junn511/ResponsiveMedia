import time
import pandas as pd
import numpy as np
from matplotlib.pylab import plt
import random
import peak_example
import math
import string
import matplotlib.animation as animation
import threading
import socket
import OSC


class JinsSocket(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        
        # set your default ip&port  
        # self.IP = socket.gethostbyname(socket.gethostname())
        self.IP = "127.0.0.1"
        self.Port = 12562
        
        """ Define the length of different mode array """
        self.FULL_COUNT = 13
        
        """ Define Array """
        self.EogL = np.zeros(0)
        self.EogR = np.zeros(0)
        self.EogH = np.zeros(0)
        self.EogV = np.zeros(0)

        self.SendingOSC = np.zeros(0)
        self.OscData = np.zeros(0)

        self.w_size = 10  

        self.count = 0   


    def setIP_Port(self, IP, Port):
        self.IP = IP
        self.Port = Port
        
    def setConnection(self):
        # setting the socket comunication
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.serverSocket.bind((self.IP, self.Port))
        self.osc = OSC.OSCClient()
        self.osc.connect((self.IP, 60000))

    
    def run(self):
        while True:
            data, address = self.serverSocket.recvfrom(2048)
            # data = self.client.recv(2048)
            converted = data.decode("utf-8")
            lines = converted.split("\r\n")
            for line in lines:
                values = list(map(int, line.split(",")))
                if len(values) == 9:
                    self.addFULL(values)
                    self.checkSIZE()
                    self.BlinkDetection()
                else:
                    break

    
    def Disconnect(self):
        self.client.close()

    """ Full Mode """
    def addFULL(self, full_values):
        self.EogL = np.append(self.EogL,full_values[7])
        self.EogR = np.append(self.EogR,full_values[8])
        self.EogH = np.append(self.EogH,full_values[7]-full_values[8])
        self.EogV = np.append(self.EogV,-(full_values[7]+full_values[8])/2)

        self.SendingOSC = np.append(self.SendingOSC,1)
        self.Sampling()

    def Sampling(self):
        if len(self.SendingOSC) == 250: #50Hz 250 5sec
            self.frequency = np.sum(self.OscData) / 5
            self.OscDataSend(self.frequency)
            print (self.frequency)
            self.frequency = np.zeros(0)
            self.OscData = np.zeros(0)
            self.SendingOSC = np.zeros(0)

        # if len(self.EogV) == 18: #100Hz 18 0.18sec
        #     self.Detection_Algorithm(self.EogH,self.EogV)
        # else:
        #     self.Buffer = np.array([0])

    def Detection_Algorithm(self,EogH_values,EogV_values):

        self.Buffer1 = np.array(EogH_values,dtype=np.int)
        self.Buffer2 = np.array(EogV_values,dtype=np.int)

        self.Vh_diff = np.diff(self.Buffer1)
        self.Vh_Sum = np.sum(np.abs(self.Vh_diff))
        self.Vh_std = np.std(self.Buffer1)

        self.Vv_diff = np.diff(self.Buffer2)
        self.Vv_Sum = np.sum(np.abs(self.Vv_diff))
        self.Vv_std = np.std(self.Buffer2)

        print 'Vv Sum: %d, Vh Sum: %d, Vh_std: %d, Vv_std: %d' % (self.Vv_Sum, self.Vh_Sum, self.Vh_std, self.Vv_std)

        # if self.Vv_std > 60:
        #     print ('blink')
        #     self.OscData = np.append(self.OscData,1)
   
        # else:
        #     print ('no')
        self.EogH = np.zeros(0)
        self.EogV = np.zeros(0)

    def BlinkDetection(self):
        Vh = np.mean(self.EogH)
        Vv = np.mean(self.EogV)
        Vh_std = np.std(self.EogH)
        Vv_std = np.std(self.EogV)

        if Vv_std > 100 and Vh_std <90:
            self.count += 1
            if self.count > 4:
                print 'blink'
                self.count = 0
                self.OscData = np.append(self.OscData,1)
        else:
            print 'no'

        print 'Vv: %d, Vh: %d, Vh_std: %d, Vv_std: %d' % (Vv, Vh, Vh_std, Vv_std)


    def OscDataSend(self,values):

        self.oscmsg = OSC.OSCMessage()
        self.oscmsg.setAddress("x")
        self.oscmsg.append(values)
        self.osc.send(self.oscmsg)


    def checkSIZE(self):
        if len(self.EogL) > self.w_size:
            self.EogL = self.EogL[-self.w_size:]
            self.EogR = self.EogR[-self.w_size:]
            self.EogH = self.EogH[-self.w_size:]
            self.EogV = self.EogV[-self.w_size:]



jins_client = JinsSocket()
jins_client.setConnection()
jins_client.start()












