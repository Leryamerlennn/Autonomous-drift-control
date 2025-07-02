import serial
import time

pathToArduino = "/dev/ttyUSB0"  
baudRate = 9600  
ser = serial.Serial(pathToArduino, baudRate)  
time.sleep(2) 

control_from_raspberry_pi = False
control_from_rc = False

def check_raspberry_pi_control():
    global control_from_raspberry_pi
    try:
        line = ser.readline().decode().strip()
        if line:  
            control_from_raspberry_pi = True
            return True
    except Exception as e:
        print(f"Error with Raspberry Pi: {e}")
    return False

def check_rc_control():
    global control_from_rc
    try:
        pwm_value = int(ser.readline().decode().strip())
        if pwm_value > 1000: 
            control_from_rc = True
            return True
    except Exception as e:
        print(f"Error with control console: {e}")
    return False

while True:
    if check_raspberry_pi_control():
        print("Robot is controlled by Raspberry Pi.")
        control_from_rc = False  
    elif check_rc_control():
        print("Robot is controlled by RC (remote control).")
        control_from_raspberry_pi = False 
    else:
        print("No active control signal detected.")
    
    time.sleep(1) 
