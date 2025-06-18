#include "TimerLib.h"

// Timer initialization
void initTimer() {
  noInterrupts();  // Disable interrupts

  // Reset registers just in case to clear any leftover bits
  TCCR1A = 0;
  TCCR1B = 0;

  /* Here we set Fast PWM mode.
     In this mode the counter counts up from 0 and resets.
     By default it can count up to 65535, but here it will count up to the value in ICR1. */

  TCCR1A |= (1 << WGM11);
  TCCR1B |= (1 << WGM12) | (1 << WGM13);

  // Prescaler set to 8 (16 MHz / 8 = 2 MHz)
  TCCR1B |= (1 << CS11);

  // Non-inverting mode for channels A and B
  TCCR1A |= (1 << COM1A1);
  TCCR1A |= (1 << COM1B1);

  // Top value (20 ms period)
  ICR1 = 39999;

  // Initial output values
  OCR1A = 2000;
  OCR1B = 2000;

  interrupts();  // Enable interrupts
}

// Set OCR1A value (motor)
void setOcr1A(uint16_t val) {
  OCR1A = val;
}

// Set OCR1B value (steering)
void setOcr1B(uint16_t val) {
  OCR1B = val;
}
