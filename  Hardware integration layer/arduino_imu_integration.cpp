#include <Arduino.h>
#include <TimerLib.h>
#include "Wire.h"
#include "I2C.h"

//#define MPU9250_IMU_ADDRESS 0x68
#define MPU9250_IMU_ADDRESS 0x69
#define MPU9250_MAG_ADDRESS 0x0C
 
#define GYRO_FULL_SCALE_250_DPS  0x00
#define GYRO_FULL_SCALE_500_DPS  0x08
#define GYRO_FULL_SCALE_1000_DPS 0x10
#define GYRO_FULL_SCALE_2000_DPS 0x18
 
#define ACC_FULL_SCALE_2G  0x00
#define ACC_FULL_SCALE_4G  0x08
#define ACC_FULL_SCALE_8G  0x10
#define ACC_FULL_SCALE_16G 0x18
 
#define INTERVAL_MS_PRINT 1
 
#define G 9.80665

float adaptiveAlpha = 0.9; 


class KalmanFilter2D {
private:
    float x[2] = {0, 0};  
    float P[2][2] = {{1, 0}, {0, 1}};  
    float Q[2] = {0.001, 0.001};  // Шум процесса (настраивается)
    float R = 0.1;              // Шум измерения (настраивается)

public:
    float update(float measurement, float gyroRate, float dt) {
        float x_pred = x[0] + dt * x[1];
        float v_pred = x[1];

        P[0][0] += dt * (dt * P[1][1] + P[0][1] + P[1][0]) + Q[0];
        P[0][1] += dt * P[1][1];
        P[1][0] += dt * P[1][1];
        P[1][1] += Q[1];
        
        float K[2];
        float S = P[0][0] + R;
        K[0] = P[0][0] / S;
        K[1] = P[1][0] / S;

        float y = measurement - x_pred;
        x[0] = x_pred + K[0] * y;
        x[1] = v_pred + K[1] * y;

        P[0][0] -= K[0] * P[0][0];
        P[0][1] -= K[0] * P[0][1];
        P[1][0] -= K[1] * P[0][0];
        P[1][1] -= K[1] * P[0][1];

        return x[0];
    }
};
 
struct gyroscope_raw {
  int16_t x, y, z;
} gyroscope;
 
struct accelerometer_raw {
  int16_t x, y, z;
} accelerometer;

struct accelerometer_Offset {
  int16_t x, y, z;
} accelerometer_offset;

struct {
  struct {
    float x, y, z;
  } accelerometer, gyroscope;
} normalized;
 
unsigned long lastPrintMillis = 0;


struct angle {
  float x, y, z = 0;
};

KalmanFilter2D kalmanPitch;
KalmanFilter2D kalmanRoll;

angle position;
angle kalmanAngles;

unsigned long lastSampleMicros = 0;


void updateAlpha(float targetAlpha, float speed = 0.1) {
    adaptiveAlpha += (targetAlpha - adaptiveAlpha) * speed;
}

bool isHardDrift(float gyroZ, float accelY) {
    return (abs(gyroZ) > 200) || (abs(accelY) > 3.0); // Пороги для гоночного авто
}

float filteredAccelX = 0;
float filteredAccelY = 0;
float filteredAccelZ = 0;
const float filterAlpha = 0.2;

void filterAccel() {
    filteredAccelX = filterAlpha * normalized.accelerometer.x + (1-filterAlpha) * filteredAccelX;
    filteredAccelY = filterAlpha * normalized.accelerometer.y + (1-filterAlpha) * filteredAccelY;
    filteredAccelZ = filterAlpha * normalized.accelerometer.z + (1-filterAlpha) * filteredAccelZ;
}


bool motorInitDone = false;  
unsigned long startTime;


bool readyToDrive = false;
unsigned long readyStart = 0;
uint16_t lastGoodServo = 3000;
uint16_t lastGoodMotor = 700;

void setup() {
  pinMode(2,INPUT);
  pinMode(3,INPUT);
  pinMode(4,INPUT);
  pinMode(5,INPUT);
  pinMode(9, OUTPUT);
  pinMode(10,OUTPUT);

  initTimer(); 
  Serial.begin(115200);
  Serial.setTimeout(10);
  setOcr1A(700);
  setOcr1B(3000); 
  delay(1000);


  //Initialize IMU
  Wire.begin();
  
 
  I2CwriteByte(MPU9250_IMU_ADDRESS, 27, GYRO_FULL_SCALE_1000_DPS); // Configure gyroscope range
  I2CwriteByte(MPU9250_IMU_ADDRESS, 28, ACC_FULL_SCALE_2G);        // Configure accelerometer range
 
  I2CwriteByte(MPU9250_IMU_ADDRESS, 56, 0x01); // Enable interrupt pin for raw data
}


void loop() {
  
  static String serialBuffer = "";
  static bool gotSerialSteer = false;
  static unsigned long lastSerialTime = 0;

  // ---------- read serial ----------
  // String s = Serial.readString();
  // int value = s.toInt();
  int valueServo = Serial.parseInt();  // считывает до первой запятой
  int valueMotor = Serial.parseInt(); 
  
  if (valueServo > 100 && valueMotor > 100) {
    lastGoodServo = valueServo;
    lastGoodMotor = valueMotor;
    
  }

  
  gotSerialSteer = true;
  lastSerialTime = millis();
  


  // // ===== СТАНДАРТНАЯ ИНИЦИАЛИЗАЦИЯ =====
  // if (!readyToDrive) {
  //   if (millis() - readyStart > 1000) {
  //     readyToDrive = true;
  //   } else {
  //     return;
  //   }
  // }

  // ===== ЧТЕНИЕ PWM =====
  unsigned long pwmValueSOS   = pulseIn(2, HIGH);
  //unsigned long pwmValueServo = pulseIn(3, HIGH);
  //unsigned long pwmValueMotor = pulseIn(4, HIGH);
  //unsigned long pwmValueAuto  = pulseIn(5, HIGH);



  
  // if (pwmValueMotor > 0 && pwmValueServo > 0) {
  //   lastGoodMotor = pwmValueMotor * 2;
  //   lastGoodServo = pwmValueServo * 2;
  // }

  // ===== ПОДАЕМ УПРАВЛЕНИЕ НА МОТОР И РУЛЬ =====
  if (pwmValueSOS >= 1459) {
    setOcr1A(700);  // тормоз
    
  } else {
    setOcr1B(lastGoodServo);    
    setOcr1A(lastGoodMotor);    
  }

  // ===== processing IMU =====

  if (!readSample()) return;
  unsigned long currentMillis = millis();
  if (currentMillis - lastPrintMillis > INTERVAL_MS_PRINT) {
    angle accAngles = calculateAccelerometerAngles();
    float accPitch = degrees(accAngles.x);
    float accRoll  = degrees(accAngles.y);

    // ===== ОТПРАВКА В SERIAL ДЛЯ RASPBERRY =====
    Serial.print(millis()); Serial.print(",");
    Serial.print(filteredAccelX, 4); Serial.print(",");
    Serial.print(filteredAccelY, 4); Serial.print(",");
    Serial.print(normalized.gyroscope.z, 4); Serial.print(",");
    Serial.print(lastGoodServo); Serial.print(",");
    Serial.print(lastGoodMotor); 
    Serial.println();

    lastPrintMillis = currentMillis;
  }

  // ===== АДАПТИВНЫЙ ФИЛЬТР =====
  // if (isHardDrift(normalized.gyroscope.z, normalized.accelerometer.y)) {
  //   updateAlpha(0.99); // упор на гироскоп
  // } else {
  //   updateAlpha(0.9);  // баланс с акселерометром
  // }
}



bool isImuReady()
{
  uint8_t isReady;

  I2Cread(MPU9250_IMU_ADDRESS, 58, 1, &isReady);

  return isReady & 0x01;
}

angle calculateAccelerometerAngles()
{
  angle accelerometer;

  accelerometer.x = atan(normalized.accelerometer.y / sqrt(sq(normalized.accelerometer.x) + sq(normalized.accelerometer.z)));
  accelerometer.y = atan(-1 * normalized.accelerometer.x / sqrt(sq(normalized.accelerometer.y) + sq(normalized.accelerometer.z)));
  accelerometer.z = atan2(accelerometer.y, accelerometer.x);

  return accelerometer;
}

angle calculateGyroscopeAngles(unsigned long sampleMicros)
{
  angle gyroscope;

  gyroscope.x = normalized.gyroscope.x * sampleMicros / 1000000;
  gyroscope.y = normalized.gyroscope.y * sampleMicros / 1000000;
  gyroscope.z = normalized.gyroscope.z * sampleMicros / 1000000;

  return gyroscope;
}

double getPitch()
{
  return position.x;
}

double getRoll()
{
  return position.y;
}



void readRawImu()
{
  uint8_t buff[14];

  // Read output registers:
  // [59-64] Accelerometer
  // [65-66] Temperature
  // [67-72] Gyroscope
  I2Cread(MPU9250_IMU_ADDRESS, 59, 14, buff);

  // Accelerometer, create 16 bits values from 8 bits data
  accelerometer.x = (buff[0] << 8 | buff[1]);
  accelerometer.y = (buff[2] << 8 | buff[3]);
  accelerometer.z = (buff[4] << 8 | buff[5]);

  // Gyroscope, create 16 bits values from 8 bits data
  gyroscope.x = (buff[8] << 8 | buff[9]);
  gyroscope.y = (buff[10] << 8 | buff[11]);
  gyroscope.z = (buff[12] << 8 | buff[13]);
}


void normalize(gyroscope_raw gyroscope)
{
  // Sensitivity Scale Factor (MPU datasheet page 8)
  normalized.gyroscope.x = gyroscope.x / 32.8;
  normalized.gyroscope.y = gyroscope.y / 32.8;
  normalized.gyroscope.z = gyroscope.z / 32.8;
}

void normalize(accelerometer_raw accelerometer)
{
//   Sensitivity Scale Factor (MPU datasheet page 9)
  normalized.accelerometer.x = (accelerometer.x) * G / 16384;
  normalized.accelerometer.y = (accelerometer.y) * G / 16384;
  normalized.accelerometer.z = (accelerometer.z) * G / 16384;
}

bool readSample() {
    if (isImuReady() == false) {
        return false;
    }

    unsigned long sampleMicros = (lastSampleMicros > 0) ? micros() - lastSampleMicros : 0;
    lastSampleMicros = micros();
    float dt = sampleMicros / 1e6f;

    readRawImu();
    normalize(gyroscope);
    normalize(accelerometer);

    angle accelerometerAngles = calculateAccelerometerAngles();
    angle gyroscopeAngles = calculateGyroscopeAngles(sampleMicros);

    filterAccel();

    kalmanAngles.x = kalmanPitch.update(
        degrees(accelerometerAngles.x), 
        normalized.gyroscope.y, 
        dt
    );
    kalmanAngles.y = kalmanRoll.update(
        degrees(accelerometerAngles.y), 
        normalized.gyroscope.x, 
        dt
    );

    position.x = adaptiveAlpha * (position.x + degrees(gyroscopeAngles.x)) + (1 - adaptiveAlpha) * kalmanAngles.x;
    position.y = adaptiveAlpha * (position.y + degrees(gyroscopeAngles.y)) + (1 - adaptiveAlpha) * kalmanAngles.y;
    return true;
}

