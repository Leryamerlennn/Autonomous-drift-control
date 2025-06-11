#This is a program template that will be further updated
#  when updating the types and amounts of data on the Arduino.

import serial
import time

pathToArduino = "/dev/ttyUSB0"
baudRate = 9600
ser = serial.Serial(pathToArduino, baudRate)
time.sleep(2)

while True:
    data = ser.readline()[:-2] #Example types: yaw, velX, velY
    dataArray = list(map(float, data.split()))
    print(dataArray)