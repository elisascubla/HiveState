#define TINY_GSM_MODEM_SIM800
#include <TinyGsmClient.h>
#include <ArduinoHttpClient.h>
#include "MHZ19.h"
#include "SoftwareSerial.h"
#include <Ascolto_e_classificazione_in_situ_di_audio_per_il_rilevamento_dello_stato_di_salute_di_un_alveare_inferencing.h>
#include <RTClib.h>
#include <ArduinoJson.h>

#define MODEM_RX 2
#define MODEM_TX 3
SoftwareSerial SerialAT(MODEM_RX, MODEM_TX);

#define GSM_PIN ""
const char apn[] = "iot.1nce.net";
const char gprsUser[] = "";
const char gprsPass[] = "";

const char* SERVER = "api.openweathermap.org";
const String API_KEY = "TUO_API_KEY";
const String LAT = "45.0";
const String LON = "9.0";

const char* THINGSPEAK_SERVER = "api.thingspeak.com";
const char* THINGSPEAK_API_KEY = "TUO_THINGSPEAK_WRITE_API_KEY";

TinyGsm modem(SerialAT);
TinyGsmClient client(modem);
HttpClient http(client, SERVER, 80);
HttpClient thingspeak(client, THINGSPEAK_SERVER, 80);

#define CO2_RX_PIN 6
#define CO2_TX_PIN 7
SoftwareSerial mySerial(CO2_RX_PIN, CO2_TX_PIN);
MHZ19 co2Sensor;

RTC_DS3231 rtc;

static bool debug_nn = false;
typedef struct {
  int16_t *buffer;
  uint8_t buf_ready;
  uint32_t buf_count;
  uint32_t n_samples;
} inference_t;

static inference_t inference;

#define BUFFER_SIZE 10
float tempBuffer[BUFFER_SIZE] = {0};
float co2Buffer[BUFFER_SIZE] = {0};

float temperatureInternal, humidity, co2Concentration;
float tempExternal, rainLast24h;
float maxTempFluctuationLast24h, maxCO2FluctuationLast24h;

struct HiveState {
    float minHumidity, maxHumidity;
    float minTempExternal, maxTempExternal;
    bool rain;
    float minTempFluctuation, maxTempFluctuation;
    float minTempInternal, maxTempInternal;
    float maxCO2Fluctuation;
    float minCO2, maxCO2;
    String label;
};

HiveState states[] = {
    {70, 95, 9, 35, false, 5, 20, 10, 37, 15, 440, 500, "X1"},
    {70, 95, -99, 9, false, 5, 20, -99, 10, 15, 440, 500, "X2"},
    {95, 999, 9, 35, false, 5, 20, 10, 37, 15, 440, 500, "X3"},
    {0, 70, 9, 35, false, 5, 20, 10, 37, 15, 440, 500, "X4"},
    {70, 95, 9, 35, false, 0, 5, 10, 37, 15, 440, 500, "X5"},
    {70, 95, 9, 35, false, 20, 999, 10, 37, 15, 440, 500, "X6"},
    {70, 95, 9, 35, false, 5, 20, 10, 37, 15, 440, 500, "X7"},
    {70, 95, 9, 35, false, 5, 20, 10, 37, 15, 0, 440, "X8"},
    {70, 95, 9, 35, false, 5, 20, 37, 999, 15, 440, 500, "X9"},
    {95, 999, 9, 35, true, 5, 20, 10, 37, 15, 440, 500, "X10"}
};

float movingAverage(float newValue, float* history, int size) {
    static int index = 0;
    static float sum = 0;
    sum -= history[index];
    history[index] = newValue;
    sum += newValue;
    index = (index + 1) % size;
    return sum / size;
}

float parseJsonFloat(const String& json, const String& key) {
    StaticJsonDocument<1024> doc;
    DeserializationError error = deserializeJson(doc, json);
    if (error) {
        Serial.print("JSON parsing failed: ");
        Serial.println(error.c_str());
        return NAN;
    }

    if (!doc.containsKey("main")) return NAN;
    if (key == "temp") return doc["main"]["temp"];

    if (key == "rain") {
        if (doc.containsKey("rain")) {
            if (doc["rain"].containsKey("1h")) return doc["rain"]["1h"];
            if (doc["rain"].containsKey("3h")) return doc["rain"]["3h"];
        }
        return 0.0;
    }
    return NAN;
}

void getWeatherData() {
    String url = "/data/2.5/weather?lat=" + LAT + "&lon=" + LON + "&appid=" + API_KEY + "&units=metric";
    http.get(url);
    int statusCode = http.responseStatusCode();
    if (statusCode == 200) {
        String response = http.responseBody();
        tempExternal = parseJsonFloat(response, "temp");
        rainLast24h = parseJsonFloat(response, "rain") > 5;
    } else {
        Serial.println("Errore richiesta meteo");
    }
}

String classifyHiveState() {
    for (auto state : states) {
        if (humidity >= state.minHumidity && humidity <= state.maxHumidity &&
            tempExternal >= state.minTempExternal && tempExternal <= state.maxTempExternal &&
            rainLast24h == state.rain &&
            maxTempFluctuationLast24h >= state.minTempFluctuation && maxTempFluctuationLast24h <= state.maxTempFluctuation &&
            temperatureInternal >= state.minTempInternal && temperatureInternal <= state.maxTempInternal &&
            maxCO2FluctuationLast24h < state.maxCO2Fluctuation &&
            co2Concentration >= state.minCO2 && co2Concentration <= state.maxCO2) {
            return state.label;
        }
    }
    return "Stato non definito";
}

static bool microphone_inference_start(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (inference.buffer == NULL) {
        return false;
    }
    inference.n_samples = n_samples;
    inference.buf_ready = 0;
    return true;
}

static bool microphone_inference_record(void) {
    for (size_t i = 0; i < inference.n_samples; i++) {
        inference.buffer[i] = analogRead(A0) - 512;
    }
    inference.buf_ready = 1;
    return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = (float)inference.buffer[offset + i] / 32768.0f;
    }
    return 0;
}

void audioClassification() {
    const uint32_t maxSamples = 16000; // ~10s @ 1600Hz, ridotto per contenere memoria
    if (!microphone_inference_start(maxSamples)) {
        Serial.println("Errore inizializzazione microfono");
        return;
    }
    if (!microphone_inference_record()) {
        Serial.println("Errore registrazione audio");
        return;
    }

    signal_t signal;
    signal.total_length = maxSamples;
    signal.get_data = &microphone_audio_signal_get_data;

    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        Serial.print("Errore modello: ");
        Serial.println(r);
        return;
    }

    Serial.println("Risultati classificazione audio:");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        Serial.print(result.classification[ix].label);
        Serial.print(": ");
        Serial.println(result.classification[ix].value, 5);
    }
    #if EI_CLASSIFIER_HAS_ANOMALY == 1
        Serial.print("Anomalia: ");
        Serial.println(result.anomaly, 3);
    #endif
    free(inference.buffer);
}

void sendToThingSpeak(String hiveState) {
    String url = "/update?api_key=" + String(THINGSPEAK_API_KEY) + "&field1=" + hiveState;
    thingspeak.get(url);
    Serial.print("Invio a ThingSpeak: ");
    Serial.println(thingspeak.responseStatusCode());
    Serial.println(thingspeak.responseBody());
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    SerialAT.begin(9600);
    delay(3000);
    Wire.begin();
    rtc.begin();

    modem.restart();
    if (GSM_PIN && modem.getSimStatus() != 3) modem.simUnlock(GSM_PIN);
    if (!modem.waitForNetwork()) {
        Serial.println("Errore rete GSM");
        return;
    }
    if (!modem.gprsConnect(apn, gprsUser, gprsPass)) {
        Serial.println("Errore connessione GPRS");
        return;
    }
    Serial.println("GPRS Connesso!");

    mySerial.begin(9600);
    co2Sensor.begin(mySerial);
    co2Sensor.autoCalibration(false);
}

void loop() {
    temperatureInternal = movingAverage(random(10, 37), tempBuffer, BUFFER_SIZE);
    humidity = random(70, 95);
    co2Concentration = movingAverage(random(440, 500), co2Buffer, BUFFER_SIZE);
    getWeatherData();

    String hiveState = classifyHiveState();
    Serial.println("Stato Alveare: " + hiveState);
    sendToThingSpeak(hiveState);

    DateTime now = rtc.now();
    int hour = now.hour();
    int month = now.month();

    bool useExternal = false;
    if ((month == 7 || month == 8) &&
        ((hour >= 9 && hour < 11) || (hour >= 16 && hour < 19)) &&
        tempExternal >= 5 && tempExternal <= 33 &&
        humidity >= 40 && humidity <= 100) {
        useExternal = true;
    } else if ((month == 5 || month == 6 || month == 9) &&
               (hour >= 11 && hour < 16) &&
               tempExternal >= 15 && tempExternal <= 25 &&
               humidity > 60) {
        useExternal = true;
    }

    if (useExternal) {
        Serial.println("Classificazione con microfono esterno (10s ogni 5 min)");
        audioClassification();
        delay(300000);
    } else if ((month == 4 || month == 5) && hour >= 11 && hour < 15) {
        Serial.println("Classificazione con microfono interno (10s ogni 10 min)");
        audioClassification();
        delay(600000);
    } else if (hour >= 7 && hour < 15) {
        Serial.println("Classificazione con microfono interno (10s ogni 10 min)");
        audioClassification();
        delay(600000);
    } else {
        delay(600000);
    }
}
