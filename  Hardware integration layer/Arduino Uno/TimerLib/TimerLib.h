#ifndef TIMER_LIB_H
#define TIMER_LIB_H

#include <Arduino.h>

// Функция инициализации Timer
void initTimer();

// Функции для установки OCR1A и OCR1B
void setOcr1A(uint16_t val);
void setOcr1B(uint16_t val);

#endif

