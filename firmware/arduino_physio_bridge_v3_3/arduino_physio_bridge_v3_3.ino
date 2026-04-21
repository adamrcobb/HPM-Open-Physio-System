// ============================================================================
// ARDUINO PHYSIO BRIDGE  v3.3
// Hardware:
//   GSR  — CJMCU-6107  →  ADS1115 AIN0  (I2C: SDA→A4, SCL→A5, ADDR→GND)
//   ECG  — Olimex SHIELD-EKG-EMG → A1   (AIN_SEL jumper position 2)
//   MCU  — Arduino Uno R3
//
// CJMCU-6107 circuit topology (confirmed by 5.14V open-circuit reading):
//
//   VCC ──[ R_ref 100kΩ ]──┬── SIG out → ADS1115 AIN0
//                          │
//                      [ R_skin / fingers ]
//                          │
//                         GND
//
//   Electrodes not connected  → V_out = VCC (~5.14V)   ← open circuit
//   Fingers on pads           → V_out DROPS toward 0V  (more conductance = lower V)
//   Electrodes shorted        → V_out = 0V
//
//   VCC is auto-measured from Arduino's internal 1.1V bandgap at startup.
//   No hardcoded supply voltage needed.
//
// Serial output format (v3.3):
//   {ms},{rawGSR},{GSR_uS},{rawECG},{ECG_mV}[,{marker}]
//   No leading 'T', no duplicate timestamp field.
//   Marker field is omitted entirely when no marker is active.
//
// Changes from v3.2:
//   - Removed leading 'T' character and duplicate ms field from data lines
//   - Marker field only appended when non-empty (no trailing comma)
//   - F() macros on all setup() strings → frees ~400 bytes of RAM
//   - Keepalive LED blink every 25 s when idle → prevents powerbank shutoff
//   - Version bump to v3.3
//
// Required libraries: Adafruit ADS1X15, Adafruit BusIO
// ============================================================================


#include <Wire.h>
#include <Adafruit_ADS1X15.h>


// ── Pin assignments ──────────────────────────────────────────────────────────
#define ECG_PIN   A1
#define TTL_PIN    2
#define LED_PIN   13


// ── ADS1115 ──────────────────────────────────────────────────────────────────
Adafruit_ADS1115 ads;
const float ADS_MV_PER_BIT = 0.1875f;   // GAIN_TWOTHIRDS: ±6.144V, 0.1875 mV/LSB


// ── GSR conversion constants ─────────────────────────────────────────────────
const float GSR_REF_OHMS = 100000.0f;   // 100 kΩ reference resistor
float       GSR_VCC      = 5.0f;        // Updated at startup by readArduinoVCC()


// ── ECG conversion ───────────────────────────────────────────────────────────
const float ECG_VREF_MV = 5000.0f;
const float ECG_BIAS_MV = 2500.0f;


// ── Timing ───────────────────────────────────────────────────────────────────
const unsigned long SAMPLE_US          = 4000UL;   // 250 Hz
const unsigned long MARKER_PERSIST_MS  = 100UL;
const unsigned long TTL_PULSE_US       = 5000UL;
const unsigned long KEEPALIVE_MS       = 25000UL;  // powerbank keepalive interval


char          currentMarker[64] = "";
unsigned long markerEndMs       = 0;
unsigned long lastSampleUs      = 0;
unsigned long lastKeepaliveMs   = 0;
bool          recording         = false;



// ============================================================================
// Read actual Arduino VCC using internal 1.1V bandgap reference.
// Works on Uno/Nano/Mega — no external components needed.
// Returns supply voltage in volts (typically 4.9–5.2V on USB power).
// ============================================================================
float readArduinoVCC() {
  ADMUX = _BV(REFS0) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
  delay(4);
  ADCSRA |= _BV(ADSC);
  while (bit_is_set(ADCSRA, ADSC));
  long raw = ADCL | (ADCH << 8);
  return 1125300L / raw / 1000.0f;
}


// ============================================================================
// GSR conversion — CJMCU-6107 standard voltage-divider topology:
//   V_out = VCC × R_skin / (R_ref + R_skin)
//   R_skin = R_ref × V_out / (VCC − V_out)
//   GSR_µS = 1,000,000 / R_skin
// ============================================================================
float adsRawToVolts(int16_t raw) {
  return max(0.0f, (float)raw * ADS_MV_PER_BIT / 1000.0f);
}


float voltsToGSR_uS(float V_out) {
  if (V_out > (GSR_VCC - 0.100f)) return 0.0f;  // Near VCC = electrodes disconnected
  if (V_out < 0.010f)             return 0.0f;  // Near 0V  = short circuit
  float denom = GSR_VCC - V_out;
  if (denom < 0.001f) return 0.0f;
  float R_skin = GSR_REF_OHMS * V_out / denom;
  if (R_skin < 1.0f) return 0.0f;
  return 1000000.0f / R_skin;
}


float adcToECG_mV(int raw) {
  return ((float)raw / 1023.0f) * ECG_VREF_MV - ECG_BIAS_MV;
}


void fireTTL(const char* label) {
  digitalWrite(TTL_PIN, HIGH);
  delayMicroseconds(TTL_PULSE_US);
  digitalWrite(TTL_PIN, LOW);
  Serial.print(F("TTLPULSESENT,"));
  Serial.println(label);
}


void handleIncomingSerial() {
  static char    buf[64];
  static uint8_t idx = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (idx > 0) {
        buf[idx] = '\0'; idx = 0;
        strncpy(currentMarker, buf, sizeof(currentMarker) - 1);
        currentMarker[sizeof(currentMarker) - 1] = '\0';
        markerEndMs = millis() + MARKER_PERSIST_MS;
        fireTTL(currentMarker);
        if (strstr(currentMarker, "experiment_start")) { recording = true;  digitalWrite(LED_PIN, HIGH); }
        if (strstr(currentMarker, "experiment_end"))   { recording = false; digitalWrite(LED_PIN, LOW);  }
      }
    } else if (idx < sizeof(buf) - 1) {
      buf[idx++] = c;
    }
  }
}



// ============================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  pinMode(TTL_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(TTL_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  // ── Measure actual VCC ────────────────────────────────────────────────────
  GSR_VCC = readArduinoVCC();

  // ── Initialise ADS1115 ────────────────────────────────────────────────────
  ads.setGain(GAIN_TWOTHIRDS);
  ads.setDataRate(RATE_ADS1115_475SPS);
  if (!ads.begin()) {
    Serial.println(F("# FATAL: ADS1115 not found!"));
    Serial.println(F("#   Check: SDA->A4  SCL->A5  VDD->5V  GND->GND  ADDR->GND"));
    while (true) { digitalWrite(LED_PIN, !digitalRead(LED_PIN)); delay(200); }
  }

  ads.startADCReading(ADS1X15_REG_CONFIG_MUX_SINGLE_0, /*continuous=*/true);
  delay(10);

  // ── Print header ──────────────────────────────────────────────────────────
  Serial.println(F("# ============================================================"));
  Serial.println(F("# ARDUINO PHYSIO BRIDGE  v3.3"));
  Serial.println(F("# GSR  : CJMCU-6107 -> ADS1115 AIN0 (standard topology)"));
  Serial.println(F("#         Electrodes off  = high voltage (~VCC)"));
  Serial.println(F("#         Fingers on pads = voltage DROPS with conductance"));
  Serial.print(  F("# VCC  : ")); Serial.print(GSR_VCC, 3); Serial.println(F("V  (measured from internal bandgap)"));
  Serial.println(F("# ECG  : Olimex SHIELD-EKG-EMG -> A1"));
  Serial.println(F("# Rate : 250 Hz"));
  Serial.println(F("# Format: {ms},{rawGSR},{GSR_uS},{rawECG},{ECG_mV}[,{marker}]"));
  Serial.println(F("# ============================================================"));

  // ── Startup check ─────────────────────────────────────────────────────────
  int16_t gsrRaw   = ads.getLastConversionResults();
  float   gsrVolts = adsRawToVolts(gsrRaw);
  float   gsrUS    = voltsToGSR_uS(gsrVolts);

  Serial.print(F("# GSR startup: raw=")); Serial.print(gsrRaw);
  Serial.print(F("  V="));               Serial.print(gsrVolts, 3);
  Serial.print(F("V  gsr="));            Serial.print(gsrUS, 2);
  Serial.println(F(" uS"));

  if (gsrVolts > GSR_VCC - 0.1f) {
    Serial.println(F("# Note: near VCC = electrodes disconnected. Attach to see signal."));
    Serial.print(  F("#   With fingers expect V to drop to ~"));
    float V_10uS = GSR_VCC * 100000.0f / (100000.0f + 100000.0f);
    Serial.print(V_10uS, 2);
    Serial.println(F("V (10 uS resting estimate)"));
  } else if (gsrVolts < 0.05f) {
    Serial.println(F("# WARNING: near 0V = electrode leads shorted?"));
  } else {
    Serial.print(F("# GSR live: ")); Serial.print(gsrUS, 1); Serial.println(F(" uS"));
  }

  Serial.println(F("# Ready. Waiting for 'experiment_start' marker."));
  lastSampleUs    = micros();
  lastKeepaliveMs = millis();
}



// ============================================================================
void loop() {
  handleIncomingSerial();

  // ── Powerbank keepalive: brief LED blink when idle ────────────────────────
  if (!recording) {
    unsigned long nowMs = millis();
    if (nowMs - lastKeepaliveMs >= KEEPALIVE_MS) {
      lastKeepaliveMs = nowMs;
      digitalWrite(LED_PIN, HIGH);
      delay(50);
      digitalWrite(LED_PIN, LOW);
    }
  }

  // ── 250 Hz sample timing ──────────────────────────────────────────────────
  unsigned long nowUs = micros();
  if (nowUs - lastSampleUs < SAMPLE_US) return;
  if (nowUs - lastSampleUs > 2UL * SAMPLE_US) lastSampleUs = nowUs;
  else                                          lastSampleUs += SAMPLE_US;

  int16_t rawGSR = ads.getLastConversionResults();
  int     rawECG = analogRead(ECG_PIN);

  float gsr_uS = voltsToGSR_uS(adsRawToVolts(rawGSR));
  float ecg_mV = adcToECG_mV(rawECG);

  // ── Serial output (v3.3 format) ───────────────────────────────────────────
  // Format: {ms},{rawGSR},{GSR_uS},{rawECG},{ECG_mV}[,{marker}]
  // Marker field omitted when no marker is active (no trailing comma).
  unsigned long ms = millis();
  Serial.print(ms);
  Serial.print(','); Serial.print(rawGSR);
  Serial.print(','); Serial.print(gsr_uS, 3);
  Serial.print(','); Serial.print(rawECG);
  Serial.print(',');
  if (millis() <= markerEndMs && currentMarker[0] != '\0') {
    Serial.print(ecg_mV, 3);
    Serial.print(','); Serial.println(currentMarker);
  } else {
    Serial.println(ecg_mV, 3);
  }
}
