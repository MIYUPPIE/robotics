#include <SoftwareSerial.h>
// Motor pins
#define IN1 8
#define IN2 9
#define IN3 10
#define IN4 11
#define ENA 5
#define ENB 6
// Ultrasonic pins
#define TRIG 3
#define ECHO 2
// SoftwareSerial to ESP32
SoftwareSerial espSerial(4, 7); // RX, TX
long duration;
int distance;
void setup() {
  Serial.begin(9600);
  espSerial.begin(9600);
  // Motor pins
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  // Ultrasonic pins
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  // Stop motors at start
  stopMotors();
}
void loop() {
  // 1. Read distance
  distance = readUltrasonic();
  // Send distance to ESP32
  espSerial.print("DIST:");
  espSerial.println(distance);
  // 2. Read command from ESP32
  if (espSerial.available()) {
    char cmd = espSerial.read();
    handleCommand(cmd);
  }
  delay(200);
}
// --- Functions ---
int readUltrasonic() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  
  duration = pulseIn(ECHO, HIGH);
  int cm = duration * 0.034 / 2;
  Serial.print("distance: ");
  Serial.println(cm);
  return cm;
}
void handleCommand(char cmd) {
  switch (cmd) {
    case 'F': forward(); break;
    case 'B': backward(); break;
    case 'L': turnLeft(); break;
    case 'R': turnRight(); break;
    case 'S': stopMotors(); break;
  }
}
// Motor movements
void forward() {
  Serial.println("Forward");
  analogWrite(ENA, 150); analogWrite(ENB, 150);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}
void backward() {
  Serial.println("Backward");
  analogWrite(ENA, 150); analogWrite(ENB, 150);
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}
void turnLeft() {
  Serial.println("Left");
  analogWrite(ENA, 150); analogWrite(ENB, 150);
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}
void turnRight() {
  Serial.println("Right");
  analogWrite(ENA, 150); analogWrite(ENB, 150);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}
void stopMotors() {
  Serial.println("Stop");
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  analogWrite(ENA, 0); analogWrite(ENB, 0);
}