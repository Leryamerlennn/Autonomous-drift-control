#include <Arduino.h>
#include <TimerLib.h>
#include "Wire.h"
#include "I2C.h"

//#define MPU9250_IMU_ADDRESS 0x68
#define MPU9250_IMU_ADDRESS 0x69
#define MPU9250_MAG_ADDRESS 0x0C

// Gyroscope full scale settings
#define GYRO_FULL_SCALE_250_DPS   0x00
#define GYRO_FULL_SCALE_500_DPS   0x08
#define GYRO_FULL_SCALE_1000_DPS  0x10
#define GYRO_FULL_SCALE_2000_DPS  0x18

// Accelerometer full scale settings
#define ACC_FULL_SCALE_2G   0x00
#define ACC_FULL_SCALE_4G   0x08
#define ACC_FULL_SCALE_8G   0x10
#define ACC_FULL_SCALE_16G  0x18

#define INTERVAL_MS_PRINT 1
#define G 9.80665  // Gravity constant

float adaptiveAlpha = 0.9;

// Kalman filter for smoothing accelerometer and gyroscope data
class KalmanFilter2D {
private:
    float x[2] = {0, 0};  // State: [angle, angular_velocity]
    float P[2][2] = {{1, 0}, {0, 1}};  // Covariance matrix
    float Q[2] = {0.001, 0.001};  // Process noise
    float R = 0.1;                // Measurement noise

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

// Raw sensor structures
struct gyroscope_raw {
    int16_t x, y, z;
} gyroscope;

struct accelerometer_raw {
    int16_t x, y, z;
} accelerometer;

struct accelerometer_Offset {
    int16_t x, y, z;
} accelerometer_offset;

// Normalized values
struct {
    struct {
        float x, y, z;
    } accelerometer, gyroscope;
} normalized;

unsigned long lastPrintMillis = 0;

// Angles in degrees
struct angle {
    float x, y, z = 0;
};

KalmanFilter2D kalmanPitch;
KalmanFilter2D kalmanRoll;

angle position;
angle kalmanAngles;

unsigned long lastSampleMicros = 0;

// Adaptive alpha update
void updateAlpha(float targetAlpha, float speed = 0.1) {
    adaptiveAlpha += (targetAlpha - adaptiveAlpha) * speed;
}

// Detect aggressive motion (for vehicles)
bool isHardDrift(float gyroZ, float accelY) {
    return (abs(gyroZ) > 200) || (abs(accelY) > 3.0);
}

// Accelerometer low-pass filter
float filteredAccelX = 0;
float filteredAccelY = 0;
float filteredAccelZ = 0;
const float filterAlpha = 0.2;

void filterAccel() {
    filteredAccelX = filterAlpha * normalized.accelerometer.x + (1 - filterAlpha) * filteredAccelX;
    filteredAccelY = filterAlpha * normalized.accelerometer.y + (1 - filterAlpha) * filteredAccelY;
    filteredAccelZ = filterAlpha * normalized.accelerometer.z + (1 - filterAlpha) * filteredAccelZ;
}

bool motorInitDone = false;
unsigned long startTime;

bool readyToDrive = false;
unsigned long readyStart = 0;
uint16_t lastGoodServo = 3000;
uint16_t lastGoodMotor = 700;

void setup() {
    pinMode(2, INPUT);
    pinMode(3, INPUT);
    pinMode(4, INPUT);
    pinMode(5, INPUT);
    pinMode(9, OUTPUT);
    pinMode(10, OUTPUT);

    initTimer();
    Serial.begin(115200);
    Serial.setTimeout(10);
    setOcr1A(700);
    setOcr1B(3000);
    delay(1000);

    // Initialize IMU
    Wire.begin();

    I2CwriteByte(MPU9250_IMU_ADDRESS, 27, GYRO_FULL_SCALE_1000_DPS);  // Set gyro range
    I2CwriteByte(MPU9250_IMU_ADDRESS, 28, ACC_FULL_SCALE_2G);         // Set accelerometer range
    I2CwriteByte(MPU9250_IMU_ADDRESS, 56, 0x01);                      // Enable interrupt for new data
}

void loop() {
    static bool gotSerialSteer = false;
    static unsigned long lastSerialTime = 0;

    // Serial input in format: servo,motor
    String line = Serial.readStringUntil('\n');
    int commaIndex = line.indexOf(',');
    if (commaIndex > 0) {
        int valueServo = line.substring(0, commaIndex).toInt();
        int valueMotor = line.substring(commaIndex + 1).toInt();

        if (valueServo > 100 && valueMotor > 100) {
            lastGoodServo = valueServo;
            lastGoodMotor = valueMotor;
        }
    }

    gotSerialSteer = true;
    lastSerialTime = millis();

    // Read PWM input (e.g. from SOS channel)
    unsigned long pwmValueSOS   = pulseIn(2, HIGH);
    //unsigned long pwmValueServo = pulseIn(3, HIGH);
    //unsigned long pwmValueMotor = pulseIn(4, HIGH);
    //unsigned long pwmValueAuto  = pulseIn(5, HIGH);

    // Control logic
    if (pwmValueSOS >= 1459) {
        setOcr1A(700);  // Brake
    } else {
        setOcr1B(lastGoodServo);
        setOcr1A(lastGoodMotor);
    }

    // IMU update and logging
    if (!readSample()) return;

    unsigned long currentMillis = millis();
    if (currentMillis - lastPrintMillis > INTERVAL_MS_PRINT) {
        angle accAngles = calculateAccelerometerAngles();
        float accPitch = degrees(accAngles.x);
        float accRoll = degrees(accAngles.y);

        Serial.print(millis()); Serial.print(",");
        Serial.print(filteredAccelX, 4); Serial.print(",");
        Serial.print(filteredAccelY, 4); Serial.print(",");
        Serial.print(normalized.gyroscope.z, 4); Serial.print(",");
        Serial.print(lastGoodServo); Serial.print(",");
        Serial.print(lastGoodMotor);
        Serial.println();

        lastPrintMillis = currentMillis;
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
    gyroscope.x = normalized.gyroscope.x * sampleMicros / 1000000.0;
    gyroscope.y = normalized.gyroscope.y * sampleMicros / 1000000.0;
    gyroscope.z = normalized.gyroscope.z * sampleMicros / 1000000.0;
    return gyroscope;
}

double getPitch() {
    return position.x;
}

double getRoll() {
    return position.y;
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
    normalized.accelerometer.x = (accelerometer.x) * G / 16384;
    normalized.accelerometer.y = (accelerometer.y) * G / 16384;
    normalized.accelerometer.z = (accelerometer.z) * G / 16384;
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
