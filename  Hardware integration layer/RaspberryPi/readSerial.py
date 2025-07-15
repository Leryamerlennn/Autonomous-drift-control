# pylint: skip-file
import serial
import time

pathToArduino = "/dev/ttyUSB0"
baudRate = 9600
ser = serial.Serial(pathToArduino, baudRate)
time.sleep(2)

while True:
    _ = ser.readline()[:-2]
    data1 = str(ser.readline()[:-2])[2:-1]
    data2 = str(ser.readline()[:-2])[2:-1]
    data3 = str(ser.readline()[:-2])[2:-1]
    data4 = str(ser.readline()[:-2])[2:-1]
    _ = ser.readline()[:-2]
    _ = ser.readline()[:-2]
    accXYZ = list(map(float, data4[24:].replace(',', '').split()))
    gyroYaw = float(data3[14:])
    print(f'dX: {accXYZ[0]}, dY: {accXYZ[1]}, dZ: {accXYZ[2]}, Yaw: {gyroYaw}')