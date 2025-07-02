#include <Arduino.h>

#define RECEIVER_PIN_2 2 
#define RECEIVER_PIN_3 3  
#define RECEIVER_PIN_4 4  
#define RECEIVER_PIN_5 5 

void setup() {

  Serial.begin(115200);

  pinMode(RECEIVER_PIN_2, INPUT);
  pinMode(RECEIVER_PIN_3, INPUT);
  pinMode(RECEIVER_PIN_4, INPUT);
  pinMode(RECEIVER_PIN_5, INPUT);

  Serial.println("Receiver Signal Test");
  delay(1000);
}

void loop() {

  long signalDuration1 = pulseIn(RECEIVER_PIN_2, HIGH);
  long signalDuration2 = pulseIn(RECEIVER_PIN_3, HIGH);
  long signalDuration3 = pulseIn(RECEIVER_PIN_4, HIGH);
  long signalDuration4 = pulseIn(RECEIVER_PIN_5, HIGH);

  if (signalDuration1 > 0) {
    Serial.print("Receiver 2 Signal Duration: ");
    Serial.print(signalDuration1);
    Serial.println(" microseconds");
  } else {
    Serial.println("Receiver 2: No signal detected");
  }

  if (signalDuration2 > 0) {
    Serial.print("Receiver 3 Signal Duration: ");
    Serial.print(signalDuration2);
    Serial.println(" microseconds");
  } else {
    Serial.println("Receiver 3: No signal detected");
  }

  if (signalDuration3 > 0) {
    Serial.print("Receiver 4 Signal Duration: ");
    Serial.print(signalDuration3);
    Serial.println(" microseconds");
  } else {
    Serial.println("Receiver 4: No signal detected");
  }

  if (signalDuration4 > 0) {
    Serial.print("Receiver 5 Signal Duration: ");
    Serial.print(signalDuration4);
    Serial.println(" microseconds");
  } else {
    Serial.println("Receiver 5: No signal detected");
  }

  delay(1000); 
}
