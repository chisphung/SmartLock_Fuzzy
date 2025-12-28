#include "cJSON.h"
#include "camera_pins.h"
#include "esp_camera.h"
#include "esp_err.h"
#include "esp_event.h"
#include "esp_idf_version.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_websocket_client.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "sensor.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define WIFI_SSID "lmaoindex"
#define WIFI_PASS "khongphaiwibu"
#define SERVER_URI "ws://10.231.182.129:8080"

// CSI Configuration
#define CSI_ENABLED 1 // Set to 1 to enable CSI data sending, 0 to disable
#define CSI_SEND_INTERVAL_MS 500 // Send CSI data every 500ms
#define CSI_BUFFER_SIZE 128      // Number of subcarriers

// WebSocket Configuration
#define WS_SEND_TIMEOUT_MS 1000 // Timeout for WebSocket sends
#define FRAME_INTERVAL_MS                                                      \
  100 // ~10 FPS target (was 50ms/20FPS - reduced for stability)
#define WS_RECONNECT_DELAY_MS 2000 // Wait before reconnecting
#define MAX_SEND_FAILURES 3        // Consecutive failures before reconnection

static const char *TAG = "ESP32CAM";
static esp_websocket_client_handle_t ws;
static EventGroupHandle_t s_wifi_event_group;

#define WIFI_CONNECTED_BIT BIT0

// CSI data storage - Double buffer to prevent race conditions
static int8_t
    csi_data_buffer[2][CSI_BUFFER_SIZE * 2]; // Double buffer for I/Q components
static volatile int csi_data_len[2] = {0, 0};
static volatile uint32_t csi_timestamp[2] = {0, 0};
static volatile int8_t csi_rssi[2] = {0, 0};
static volatile int write_buffer_index =
    0; // Which buffer the interrupt writes to (0 or 1)

/* ---------------- CSI CALLBACK ---------------- */
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf || info->len == 0) {
    return;
  }

  // Write to the current write buffer (double buffering prevents races)
  int write_idx = write_buffer_index; // Local copy for consistency
  int copy_len =
      (info->len < CSI_BUFFER_SIZE * 2) ? info->len : CSI_BUFFER_SIZE * 2;
  memcpy(csi_data_buffer[write_idx], info->buf, copy_len);
  csi_data_len[write_idx] = copy_len;
  csi_timestamp[write_idx] = esp_log_timestamp();
  csi_rssi[write_idx] = info->rx_ctrl.rssi;
}

static void csi_init(void) {
  wifi_csi_config_t csi_config = {
      .lltf_en = true,            // Enable LLTF (Long Training Field)
      .htltf_en = true,           // Enable HTLTF (HT Long Training Field)
      .stbc_htltf2_en = true,     // Enable STBC HT-LTF2
      .ltf_merge_en = true,       // Enable LTF merge
      .channel_filter_en = false, // Don't filter by channel
      .manu_scale = false,        // Don't scale manually
      .shift = 0,                 // No shift
  };

  ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
  ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&wifi_csi_rx_cb, NULL));
  ESP_ERROR_CHECK(esp_wifi_set_csi(true));

  ESP_LOGI(TAG, "CSI initialized and enabled");
}

static bool send_csi_data(void) {
  if (!ws || !esp_websocket_client_is_connected(ws)) {
    return false;
  }

  // Swap buffers atomically - read from the buffer that's NOT being written to
  int read_idx = write_buffer_index;
  write_buffer_index = 1 - write_buffer_index; // Toggle between 0 and 1

  // Now read from read_idx while interrupt writes to the other buffer
  if (csi_data_len[read_idx] == 0) {
    return false;
  }

  // Create JSON object for CSI data
  cJSON *csi_obj = cJSON_CreateObject();
  if (!csi_obj) {
    return false;
  }

  cJSON_AddStringToObject(csi_obj, "type", "csi");
  cJSON_AddNumberToObject(csi_obj, "timestamp", csi_timestamp[read_idx]);
  cJSON_AddNumberToObject(csi_obj, "rssi", csi_rssi[read_idx]);
  cJSON_AddNumberToObject(csi_obj, "len", csi_data_len[read_idx]);

  // Create array for CSI amplitude data
  cJSON *amplitudes = cJSON_CreateArray();
  if (amplitudes) {
    // Calculate amplitude from I/Q pairs: amp = sqrt(I^2 + Q^2)
    for (int i = 0; i < csi_data_len[read_idx] / 2; i++) {
      int8_t real = csi_data_buffer[read_idx][i * 2];
      int8_t imag = csi_data_buffer[read_idx][i * 2 + 1];
      float amplitude = sqrtf((float)(real * real + imag * imag));
      cJSON_AddItemToArray(amplitudes, cJSON_CreateNumber((int)amplitude));
    }
    cJSON_AddItemToObject(csi_obj, "amplitudes", amplitudes);
  }

  // Send JSON via WebSocket
  char *payload = cJSON_PrintUnformatted(csi_obj);
  bool success = false;
  if (payload) {
    int sent = esp_websocket_client_send_text(
        ws, payload, strlen(payload), pdMS_TO_TICKS(WS_SEND_TIMEOUT_MS));
    success = (sent >= 0);
    free(payload);
  }

  cJSON_Delete(csi_obj);

  return success;
}

/* ---------------- UTILITIES ---------------- */
static bool send_ws_json(cJSON *obj) {
  if (!obj) {
    ESP_LOGE(TAG, "Attempted to send null JSON object");
    return false;
  }

  if (!ws || !esp_websocket_client_is_connected(ws)) {
    ESP_LOGW(TAG, "WebSocket not connected; dropping JSON response");
    return false;
  }

  char *payload = cJSON_PrintUnformatted(obj);
  if (!payload) {
    ESP_LOGE(TAG, "Failed to encode JSON payload");
    return false;
  }

  int sent = esp_websocket_client_send_text(ws, payload, strlen(payload),
                                            pdMS_TO_TICKS(WS_SEND_TIMEOUT_MS));
  if (sent < 0) {
    ESP_LOGE(TAG, "Failed to send JSON payload over WebSocket");
    free(payload);
    return false;
  }
  free(payload);
  return true;
}

static void send_error_response(const char *message) {
  if (!message) {
    message = "Unknown JSON error";
  }

  cJSON *err = cJSON_CreateObject();
  if (!err) {
    ESP_LOGE(TAG, "Failed to allocate JSON error response");
    return;
  }

  cJSON_AddStringToObject(err, "status", "error");
  cJSON_AddStringToObject(err, "message", message);
  send_ws_json(err);
  cJSON_Delete(err);
}

/* ---------------- WIFI INIT ---------------- */
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
  if (!s_wifi_event_group) {
    return;
  }

  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
    ESP_LOGI(TAG, "Wi-Fi STA start; connecting...");
    esp_wifi_connect();
  } else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_STA_DISCONNECTED) {
    ESP_LOGW(TAG, "Wi-Fi disconnected; retrying...");
    xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    esp_wifi_connect();
  } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
    ESP_LOGI(TAG, "Wi-Fi connected, IP: " IPSTR, IP2STR(&event->ip_info.ip));
    xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
  }
}

static esp_err_t wifi_init_sta(void) {
  s_wifi_event_group = xEventGroupCreate();
  if (!s_wifi_event_group) {
    ESP_LOGE(TAG, "Failed to create Wi-Fi event group");
    return ESP_ERR_NO_MEM;
  }

  esp_err_t err = esp_netif_init();
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_netif_init failed: %s", esp_err_to_name(err));
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }

  err = esp_event_loop_create_default();
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
    ESP_LOGE(TAG, "esp_event_loop_create_default failed: %s",
             esp_err_to_name(err));
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  } else if (err == ESP_ERR_INVALID_STATE) {
    ESP_LOGW(TAG, "Event loop already created; continuing");
  }

  if (!esp_netif_create_default_wifi_sta()) {
    ESP_LOGE(TAG, "Failed to create default Wi-Fi STA");
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return ESP_FAIL;
  }

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  err = esp_wifi_init(&cfg);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_wifi_init failed: %s", esp_err_to_name(err));
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }
  err = esp_wifi_set_mode(WIFI_MODE_STA);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_wifi_set_mode failed: %s", esp_err_to_name(err));
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }

  wifi_config_t wifi_config = {
      .sta =
          {
              .ssid = WIFI_SSID,
              .password = WIFI_PASS,
              .threshold.authmode =
                  WIFI_AUTH_OPEN, // Accept any auth mode (hotspot compatible)
          },
  };
  err = esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_wifi_set_config failed: %s", esp_err_to_name(err));
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }
  bool wifi_handler_registered = false;
  bool ip_handler_registered = false;
  err = esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                   &wifi_event_handler, NULL);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to register Wi-Fi event handler: %s",
             esp_err_to_name(err));
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }
  wifi_handler_registered = true;
  err = esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                   &wifi_event_handler, NULL);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to register IP event handler: %s",
             esp_err_to_name(err));
    if (wifi_handler_registered) {
      esp_event_handler_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                   &wifi_event_handler);
    }
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }
  ip_handler_registered = true;
  err = esp_wifi_start();
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_wifi_start failed: %s", esp_err_to_name(err));
    if (wifi_handler_registered) {
      esp_event_handler_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                   &wifi_event_handler);
    }
    if (ip_handler_registered) {
      esp_event_handler_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                   &wifi_event_handler);
    }
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }
  ESP_LOGI(TAG, "Connecting to Wi-Fi %s", WIFI_SSID);

  err = esp_wifi_connect();
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_wifi_connect failed: %s", esp_err_to_name(err));
    if (wifi_handler_registered) {
      esp_event_handler_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                   &wifi_event_handler);
    }
    if (ip_handler_registered) {
      esp_event_handler_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                   &wifi_event_handler);
    }
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }

  EventBits_t bits =
      xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE,
                          pdFALSE, pdMS_TO_TICKS(10000));

  if ((bits & WIFI_CONNECTED_BIT) == 0) {
    ESP_LOGE(TAG, "Wi-Fi connection timeout");
    if (wifi_handler_registered) {
      esp_event_handler_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                   &wifi_event_handler);
    }
    if (ip_handler_registered) {
      esp_event_handler_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                   &wifi_event_handler);
    }
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return ESP_ERR_TIMEOUT;
  }

  ESP_LOGI(TAG, "Wi-Fi station initialized successfully");
  return ESP_OK;
}

/* ---------------- CAMERA INIT ---------------- */
static void camera_init(void) {
  camera_config_t config = {
      .pin_pwdn = CAM_PIN_PWDN,
      .pin_reset = CAM_PIN_RESET,
      .pin_xclk = CAM_PIN_XCLK,
      .pin_sccb_sda = CAM_PIN_SIOD,
      .pin_sccb_scl = CAM_PIN_SIOC,
      .pin_d7 = CAM_PIN_D7,
      .pin_d6 = CAM_PIN_D6,
      .pin_d5 = CAM_PIN_D5,
      .pin_d4 = CAM_PIN_D4,
      .pin_d3 = CAM_PIN_D3,
      .pin_d2 = CAM_PIN_D2,
      .pin_d1 = CAM_PIN_D1,
      .pin_d0 = CAM_PIN_D0,
      .pin_vsync = CAM_PIN_VSYNC,
      .pin_href = CAM_PIN_HREF,
      .pin_pclk = CAM_PIN_PCLK,
      .xclk_freq_hz = 20000000,
      .ledc_timer = LEDC_TIMER_0,
      .ledc_channel = LEDC_CHANNEL_0,
      .pixel_format = PIXFORMAT_JPEG,
      .frame_size = FRAMESIZE_QVGA,
      .jpeg_quality = 20, // Higher = more compression = faster transfer (was 8)
      .fb_count = 2};     // Reduced buffer count for lower memory usage
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK)
    ESP_LOGE(TAG, "Camera init failed 0x%x", err);
  else
    ESP_LOGI(TAG, "Camera OK");
}

/* ---------------- WEBSOCKET EVENTS ---------------- */
static void on_ws_event(void *arg, esp_event_base_t base, int32_t eid,
                        void *data) {
  esp_websocket_event_data_t *event = (esp_websocket_event_data_t *)data;

  switch (eid) {
  case WEBSOCKET_EVENT_CONNECTED:
    ESP_LOGI(TAG, "WebSocket connected");
    break;
  case WEBSOCKET_EVENT_DISCONNECTED:
    ESP_LOGW(TAG, "WebSocket disconnected");
    break;
  case WEBSOCKET_EVENT_ERROR:
    if (event) {
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 2, 0)
      ESP_LOGE(TAG, "WebSocket transport error, type=%d, esp_tls=0x%x",
               event->error_handle.error_type,
               event->error_handle.esp_tls_last_esp_err);
#else
      ESP_LOGE(TAG,
               "WebSocket transport error, type=%d, esp_tls=0x%x, errno=%d",
               event->error_handle.error_type,
               event->error_handle.esp_tls_last_esp_err,
               event->error_handle.last_errno);
#endif
    } else {
      ESP_LOGE(TAG, "WebSocket transport error with no details");
    }
    break;
  case WEBSOCKET_EVENT_DATA:
    if (!event) {
      ESP_LOGW(TAG, "WebSocket data event with null payload");
      return;
    }
    if (event->op_code == WS_TRANSPORT_OPCODES_BINARY) {
      ESP_LOGD(TAG, "Binary data received (%d bytes) ignored", event->data_len);
      return;
    }
    char *json = strndup(event->data_ptr, event->data_len);
    if (!json) {
      ESP_LOGE(TAG, "Failed to allocate buffer for JSON payload");
      return;
    }
    cJSON *root = cJSON_Parse(json);
    if (!root) {
      ESP_LOGW(TAG, "Invalid JSON payload from WebSocket");
      send_error_response("Invalid JSON payload");
      free(json);
      return;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
      ESP_LOGE(TAG, "Camera sensor unavailable");
      send_error_response("Camera sensor unavailable");
      cJSON_Delete(root);
      free(json);
      return;
    }

    bool updated = false;

    const cJSON *brightness =
        cJSON_GetObjectItemCaseSensitive(root, "brightness");
    if (brightness) {
      if (!cJSON_IsNumber(brightness)) {
        ESP_LOGW(TAG, "Invalid type for brightness field");
        send_error_response("Field 'brightness' must be numeric");
        cJSON_Delete(root);
        free(json);
        return;
      }
      s->set_brightness(s, brightness->valueint);
      ESP_LOGI(TAG, "Set brightness to %d", s->status.brightness);
      updated = true;
    }

    const cJSON *contrast = cJSON_GetObjectItemCaseSensitive(root, "contrast");
    if (contrast) {
      if (!cJSON_IsNumber(contrast)) {
        ESP_LOGW(TAG, "Invalid type for contrast field");
        send_error_response("Field 'contrast' must be numeric");
        cJSON_Delete(root);
        free(json);
        return;
      }
      s->set_contrast(s, contrast->valueint);
      ESP_LOGI(TAG, "Set contrast to %d", s->status.contrast);
      updated = true;
    }

    const cJSON *saturation =
        cJSON_GetObjectItemCaseSensitive(root, "saturation");
    if (saturation) {
      if (!cJSON_IsNumber(saturation)) {
        ESP_LOGW(TAG, "Invalid type for saturation field");
        send_error_response("Field 'saturation' must be numeric");
        cJSON_Delete(root);
        free(json);
        return;
      }
      s->set_saturation(s, saturation->valueint);
      ESP_LOGI(TAG, "Set saturation to %d", s->status.saturation);
      updated = true;
    }

    const cJSON *quality = cJSON_GetObjectItemCaseSensitive(root, "quality");
    if (quality) {
      if (!cJSON_IsNumber(quality)) {
        ESP_LOGW(TAG, "Invalid type for quality field");
        send_error_response("Field 'quality' must be numeric");
        cJSON_Delete(root);
        free(json);
        return;
      }
      s->set_quality(s, quality->valueint);
      ESP_LOGI(TAG, "Set quality to %d", s->status.quality);
      updated = true;
    }

    if (!updated) {
      // Gracefully ignore JSON without recognized camera fields (e.g., acks,
      // pings)
      ESP_LOGD(TAG, "Received JSON without camera config fields, ignoring");
      cJSON_Delete(root);
      free(json);
      return;
    }

    cJSON *resp = cJSON_CreateObject();
    if (!resp) {
      ESP_LOGE(TAG, "Failed to allocate JSON response");
      cJSON_Delete(root);
      free(json);
      return;
    }
    cJSON_AddStringToObject(resp, "status", "ok");
    cJSON_AddStringToObject(resp, "message", "Camera parameters updated");
    if (send_ws_json(resp)) {
      ESP_LOGI(TAG, "Camera parameters updated and acknowledged");
    } else {
      ESP_LOGW(TAG, "Failed to deliver camera update acknowledgment");
    }
    cJSON_Delete(resp);
    cJSON_Delete(root);
    free(json);
    break;
  default:
    ESP_LOGD(TAG, "Unhandled WebSocket event id=%ld", eid);
    break;
  }
}

/* ---------------- MAIN ---------------- */
void app_main(void) {
  esp_err_t nvs_ret = nvs_flash_init();
  if (nvs_ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      nvs_ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    nvs_ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(nvs_ret);
  ESP_ERROR_CHECK(wifi_init_sta());

  // Initialize CSI after WiFi is connected
#if CSI_ENABLED
  csi_init();
#endif

  camera_init();

  esp_websocket_client_config_t ws_cfg = {.uri = SERVER_URI};
  ws = esp_websocket_client_init(&ws_cfg);
  if (!ws) {
    ESP_LOGE(TAG, "Failed to create WebSocket client");
    return;
  }
  esp_websocket_register_events(ws, WEBSOCKET_EVENT_ANY, on_ws_event, NULL);
  esp_err_t ws_start_err = esp_websocket_client_start(ws);
  if (ws_start_err != ESP_OK) {
    ESP_LOGE(TAG, "WebSocket start failed: %s", esp_err_to_name(ws_start_err));
    return;
  }
  ESP_LOGI(TAG, "WebSocket client started: %s", SERVER_URI);

  uint32_t last_csi_send = 0;
  int consecutive_failures = 0;

  while (true) {
    if (!esp_websocket_client_is_connected(ws)) {
      // Longer wait on disconnect to avoid rapid reconnect cycles
      ESP_LOGW(TAG, "WebSocket disconnected, waiting %dms before retry...",
               WS_RECONNECT_DELAY_MS);
      vTaskDelay(pdMS_TO_TICKS(WS_RECONNECT_DELAY_MS));
      consecutive_failures = 0; // Reset on reconnect
      continue;
    }

    // Send camera frame with timeout (non-blocking if network is slow)
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) {
      int sent =
          esp_websocket_client_send_bin(ws, (const char *)fb->buf, fb->len,
                                        pdMS_TO_TICKS(WS_SEND_TIMEOUT_MS));
      if (sent < 0) {
        consecutive_failures++;
        ESP_LOGW(TAG, "Frame send failed (%d/%d)", consecutive_failures,
                 MAX_SEND_FAILURES);

        if (consecutive_failures >= MAX_SEND_FAILURES) {
          ESP_LOGE(TAG, "Too many failures, forcing reconnection");
          esp_websocket_client_close(ws, pdMS_TO_TICKS(500));
          vTaskDelay(pdMS_TO_TICKS(WS_RECONNECT_DELAY_MS));
          esp_websocket_client_start(ws);
          consecutive_failures = 0;
        }
      } else {
        consecutive_failures = 0; // Reset on success
      }
      esp_camera_fb_return(fb);
    } else {
      ESP_LOGW(TAG, "Failed to get camera frame buffer");
    }

#if CSI_ENABLED
    // Send CSI data periodically
    uint32_t now = esp_log_timestamp();
    if (now - last_csi_send >= CSI_SEND_INTERVAL_MS) {
      if (send_csi_data()) {
        ESP_LOGD(TAG, "CSI data sent");
      }
      last_csi_send = now;
    }
#endif

    vTaskDelay(pdMS_TO_TICKS(FRAME_INTERVAL_MS)); // ~20 FPS when streaming
  }
}
