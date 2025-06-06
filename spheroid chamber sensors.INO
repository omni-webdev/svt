// === Sensor Pins ===
const int PRESSURE_PIN = A0;
const int HALL_PIN     = A1;
const int TEMP_PIN     = A2;

// === Constants for Conversion ===
float analogToVoltage(int raw) {
  return raw * (5.0 / 1023.0);
}

void setup() {
  Serial.begin(9600);
  delay(1000);
  Serial.println("Timestamp(ms),Pressure(V),Magnetic(V),Temp(V)");
}

void loop() {
  unsigned long ts = millis();
  float p  = analogToVoltage(analogRead(PRESSURE_PIN));
  float m  = analogToVoltage(analogRead(HALL_PIN));
  float t  = analogToVoltage(analogRead(TEMP_PIN));

  Serial.print(ts); Serial.print(",");
  Serial.print(p, 3); Serial.print(",");
  Serial.print(m, 3); Serial.print(",");
  Serial.println(t, 3);

  delay(100);  // Log every 100 ms
}
