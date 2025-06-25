#include <Arduino.h>
#include "Wire.h"
#include "I2C.h"

#define MPU9250_IMU_ADDRESS 0x68
#define GYRO_FULL_SCALE_1000_DPS 0x10
#define ACC_FULL_SCALE_2G  0x00
#define INTERVAL_MS_PRINT 100
#define G 9.80665

float adaptiveAlpha = 0.9;

class KalmanFilter2D {
private:
    float x[2] = {0, 0};  
    float P[2][2] = {{1, 0}, {0, 1}};  
    float Q[2] = {0.001, 0.001};  
    float R = 0.1;              

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

struct {
  struct {
    float x, y, z;
  } accelerometer, gyroscope;
} normalized;

struct angle {
  float x, y, z = 0;
};

KalmanFilter2D kalmanPitch;
KalmanFilter2D kalmanRoll;

angle position;
angle kalmanAngles;

unsigned long lastPrintMillis = 0;
unsigned long lastSampleMicros = 0;

float filteredAccelX = 0;
float filteredAccelY = 0;
float filteredAccelZ = 0;
const float filterAlpha = 0.2;

void updateAlpha(float targetAlpha, float speed = 0.1) {
    adaptiveAlpha += (targetAlpha - adaptiveAlpha) * speed;
}

bool isHardDrift(float gyroZ, float accelY) {
    return (abs(gyroZ) > 200) || (abs(accelY) > 3.0);
}

void filterAccel() {
    filteredAccelX = filterAlpha * normalized.accelerometer.x + (1-filterAlpha) * filteredAccelX;
    filteredAccelY = filterAlpha * normalized.accelerometer.y + (1-filterAlpha) * filteredAccelY;
    filteredAccelZ = filterAlpha * normalized.accelerometer.z + (1-filterAlpha) * filteredAccelZ;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  I2CwriteByte(MPU9250_IMU_ADDRESS, 27, GYRO_FULL_SCALE_1000_DPS);
  I2CwriteByte(MPU9250_IMU_ADDRESS, 28, ACC_FULL_SCALE_2G);
  I2CwriteByte(MPU9250_IMU_ADDRESS, 56, 0x01);
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - lastPrintMillis > INTERVAL_MS_PRINT) {
    if (!readSample()) return;

    angle accAngles = calculateAccelerometerAngles();

    float accPitch = degrees(accAngles.x);
    float accRoll = degrees(accAngles.y);

    float errorPitch = kalmanAngles.x - accPitch;
    float errorRoll  = kalmanAngles.y - accRoll;

    Serial.print(abs(errorPitch), 2);
//    Serial.print(abs(errorRoll), 2);
    Serial.println();

    lastPrintMillis = currentMillis;
  }

  if (isHardDrift(normalized.gyroscope.z, normalized.accelerometer.y)) {
    updateAlpha(0.99);
  } else {
    updateAlpha(0.9);
  }
}

bool isImuReady() {
  uint8_t isReady;
  I2Cread(MPU9250_IMU_ADDRESS, 58, 1, &isReady);
  return isReady & 0x01;
}

angle calculateAccelerometerAngles() {
  angle accelerometer;
  accelerometer.x = atan(normalized.accelerometer.y / sqrt(sq(normalized.accelerometer.x) + sq(normalized.accelerometer.z)));
  accelerometer.y = atan(-1 * normalized.accelerometer.x / sqrt(sq(normalized.accelerometer.y) + sq(normalized.accelerometer.z)));
  accelerometer.z = atan2(accelerometer.y, accelerometer.x);
  return accelerometer;
}

angle calculateGyroscopeAngles(unsigned long sampleMicros) {
  angle gyroscope;
  gyroscope.x = normalized.gyroscope.x * sampleMicros / 1000000;
  gyroscope.y = normalized.gyroscope.y * sampleMicros / 1000000;
  gyroscope.z = normalized.gyroscope.z * sampleMicros / 1000000;
  return gyroscope;
}

void readRawImu() {
  uint8_t buff[14];
  I2Cread(MPU9250_IMU_ADDRESS, 59, 14, buff);

  accelerometer.x = (buff[0] << 8 | buff[1]);
  accelerometer.y = (buff[2] << 8 | buff[3]);
  accelerometer.z = (buff[4] << 8 | buff[5]);

  gyroscope.x = (buff[8] << 8 | buff[9]);
  gyroscope.y = (buff[10] << 8 | buff[11]);
  gyroscope.z = (buff[12] << 8 | buff[13]);
}

void normalize(gyroscope_raw gyroscope) {
  normalized.gyroscope.x = gyroscope.x / 32.8;
  normalized.gyroscope.y = gyroscope.y / 32.8;
  normalized.gyroscope.z = gyroscope.z / 32.8;
}

void normalize(accelerometer_raw accelerometer) {
  normalized.accelerometer.x = accelerometer.x * G / 16384;
  normalized.accelerometer.y = accelerometer.y * G / 16384;
  normalized.accelerometer.z = accelerometer.z * G / 16384;
}

bool readSample() {
    if (!isImuReady()) return false;

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
