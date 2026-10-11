#include "esp_camera.h"
#include <WiFi.h>
// ====== WIFI SETTINGS ======
const char* ssid = "Airtel 5G CPE_83F6CD";
const char* password = "2T5SXVGUA6";
// ====== TCP SERVER ======
WiFiServer server(5000);
WiFiClient client;
// ====== SERIAL TO ARDUINO ======
HardwareSerial arduinoSerial(2);  // UART2
// ====== CAMERA PINS (AI Thinker) ======
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
void startCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  esp_camera_init(&config);
}
void setup() {
  Serial.begin(115200);
  // Serial to Arduino
  arduinoSerial.begin(9600, SERIAL_8N1, 14, 15); // RX, TX
  // Connect WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP());
  startCamera();
  server.begin();
}
void loop() {
  // Accept client
  if (!client || !client.connected()) {
    client = server.available();
  }
  // If PC sends command
  if (client && client.connected() && client.available()) {
    char cmd = client.read();
    arduinoSerial.write(cmd);  // Forward to Arduino
  }
  // If Arduino sends distance
  if (arduinoSerial.available()) {
    String data = arduinoSerial.readStringUntil('\n');
    if (client && client.connected()) {
      client.println(data);  // Send to PC
    }
  }
}