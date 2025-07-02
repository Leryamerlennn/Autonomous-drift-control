#include <Servo.h>  

#define MOTOR1_PWM_PIN 9    
#define MOTOR2_PWM_PIN 10   
#define SERVO_PIN 6         

int lastGoodServo = 3000;  
int lastGoodMotor = 3500; 

Servo steeringServo;

void setup() {
  Serial.begin(115200);   

  pinMode(MOTOR1_PWM_PIN, OUTPUT);
  pinMode(MOTOR2_PWM_PIN, OUTPUT);

  steeringServo.attach(SERVO_PIN);  
  steeringServo.write(map(lastGoodServo, 2000, 4000, 0, 180)); 
  delay(1000);  
}

void loop() {
  for (int motorSpeed = 0; motorSpeed <= 255; motorSpeed++) {
    lastGoodMotor = map(motorSpeed, 0, 255, 2886, 4002);  

    for (int angle = -90; angle <= 90; angle++) {
      lastGoodServo = map(angle, -90, 90, 2000, 4000); 

      analogWrite(MOTOR1_PWM_PIN, motorSpeed); 

      steeringServo.write(map(lastGoodServo, 2000, 4000, 0, 180));  
      Serial.print("Motor Speed: ");
      Serial.print(motorSpeed);
      Serial.print(", Steering Angle: ");
      Serial.println(angle);

      delay(100); 
    }
  }

  for (int motorSpeed = 255; motorSpeed >= 0; motorSpeed--) {
    lastGoodMotor = map(motorSpeed, 0, 255, 2886, 4002);  

    for (int angle = 90; angle >= -90; angle--) {
      lastGoodServo = map(angle, -90, 90, 2000, 4000);  

      analogWrite(MOTOR1_PWM_PIN, motorSpeed);  
      analogWrite(MOTOR2_PWM_PIN, motorSpeed);  

      steeringServo.write(map(lastGoodServo, 2000, 4000, 0, 180)); 

      Serial.print("Motor Speed: ");
      Serial.print(motorSpeed);
      Serial.print(", Steering Angle: ");
      Serial.println(angle);

      delay(100);  
    }
  }
}
