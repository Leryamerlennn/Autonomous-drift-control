# pylint: skip-file
# Send pwm signals to Arduiono using SSH connection

import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200)
time.sleep(2) 

def send_pwm(steer_pwm, throttle_pwm):
    cmd = f"{steer_pwm} {throttle_pwm}\n"
    ser.write(cmd.encode('utf-8'))
    print(f"Send: {cmd.strip()}")


send_pwm(2000, 3500)