import serial
import time

pathToArduino = "/dev/ttyUSB0"
baudRate = 9600
ser = serial.Serial(pathToArduino, baudRate)
time.sleep(2)

while True:
    _ = ser.readline()[:-2]
    data1 = ser.readline()[:-2]
    data2 = ser.readline()[:-2]
    data3 = ser.readline()[:-2]
    data4 = ser.readline()[:-2]
    _ = ser.readline()[:-2]
    _ = ser.readline()[:-2]
    data = ser.readline()[:-2]
    dataArray = list(map(float, data.split()))
    accXYZ = list(map(float, data4[23:].split(", ")))
    gyroYaw = float(data3[12:])
    print(*accXYZ, gyroYaw)