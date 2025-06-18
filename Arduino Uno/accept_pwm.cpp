#include <Arduino.h>
#include "TimerLib/TimerLib.h" 

void setup() {
  Serial.begin(115200);

  pinMode(9, OUTPUT);  
  pinMode(10, OUTPUT);  

  initTimer();  

  setOcr1B(3000);  
  setOcr1A(700); 
}

void loop() {
  if (Serial.available()) {
    int steer = Serial.parseInt();   
    int throttle = Serial.parseInt();

    if (Serial.read() == '\n') {      
      setOcr1B(steer);                
      setOcr1A(throttle);             

      Serial.print("Steer: ");
      Serial.print(steer);
      Serial.print(" | Throttle: ");
      Serial.println(throttle);
    }
  }
}
