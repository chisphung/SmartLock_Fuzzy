
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
#include "driver/gpio.h"
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/* ======================== CONFIGURATION ======================== */

#define WIFI_SSID "nhmc"
#define WIFI_PASS "14112005"
#define SERVER_URI "ws://10.0.53.62:8080"

/* Lock relay GPIO (active HIGH to unlock) */
#define LOCK_GPIO GPIO_NUM_12
#define LOCK_OPEN_DURATION_MS 3000 /* Keep unlocked for 3 seconds */

/* LED indicator GPIO (built-in LED on most ESP32-CAM boards) */
#define LED_GPIO GPIO_NUM_33

/* WebSocket settings */
#define WS_SEND_TIMEOUT_MS 600
#define FRAME_INTERVAL_MS 100 /* ~10 FPS */
#define WS_RECONNECT_DELAY_MS 500
#define MAX_SEND_FAILURES 3
#define WS_BUFFER_SIZE 32768

/* Camera recovery */
#define MAX_CAMERA_NULL_FRAMES 15

static const char *TAG = "SMARTLOCK";
static esp_websocket_client_handle_t ws;
static EventGroupHandle_t s_wifi_event_group;

/* Connection state */
static volatile bool s_ws_connected = false;
static volatile bool s_ws_started = false;

/* Lock state */
static volatile bool s_lock_open = false;
static volatile uint32_t s_lock_open_time = 0;

#define WIFI_CONNECTED_BIT BIT0

/* ======================== LOCK CONTROL ======================== */

static void lock_gpio_init(void)
{
  gpio_config_t io_conf = {
      .pin_bit_mask = (1ULL << LOCK_GPIO) | (1ULL << LED_GPIO),
      .mode = GPIO_MODE_OUTPUT,
      .pull_up_en = GPIO_PULLUP_DISABLE,
      .pull_down_en = GPIO_PULLDOWN_DISABLE,
      .intr_type = GPIO_INTR_DISABLE,
  };
  gpio_config(&io_conf);

  /* Default: locked, LED off */
  gpio_set_level(LOCK_GPIO, 0);
  gpio_set_level(LED_GPIO, 1); /* LED is active LOW on ESP32-CAM */
  ESP_LOGI(TAG, "Lock GPIO %d initialized (locked)", LOCK_GPIO);
}

static void lock_open(void)
{
  gpio_set_level(LOCK_GPIO, 1); /* Activate relay */
  gpio_set_level(LED_GPIO, 0);  /* LED ON (active LOW) */
  s_lock_open = true;
  s_lock_open_time = esp_log_timestamp();
  ESP_LOGI(TAG, "LOCK OPENED");
}

static void lock_close(void)
{
  gpio_set_level(LOCK_GPIO, 0);
  gpio_set_level(LED_GPIO, 1); /* LED OFF */
  s_lock_open = false;
  ESP_LOGI(TAG, "LOCK CLOSED");
}

/* Auto-close lock after timeout (called from main loop) */
static void lock_check_timeout(void)
{
  if (s_lock_open)
  {
    uint32_t elapsed = esp_log_timestamp() - s_lock_open_time;
    if (elapsed >= LOCK_OPEN_DURATION_MS)
    {
      lock_close();
    }
  }
}

/* ======================== UTILITIES ======================== */

static bool send_ws_json(cJSON *obj)
{
  if (!obj)
  {
    ESP_LOGE(TAG, "Attempted to send null JSON object");
    return false;
  }

  if (!ws || !esp_websocket_client_is_connected(ws))
  {
    ESP_LOGW(TAG, "WebSocket not connected; dropping JSON response");
    return false;
  }

  char *payload = cJSON_PrintUnformatted(obj);
  if (!payload)
  {
    ESP_LOGE(TAG, "Failed to encode JSON payload");
    return false;
  }

  int sent = esp_websocket_client_send_text(ws, payload, strlen(payload),
                                            pdMS_TO_TICKS(WS_SEND_TIMEOUT_MS));
  if (sent < 0)
  {
    ESP_LOGE(TAG, "Failed to send JSON payload over WebSocket");
    free(payload);
    return false;
  }
  free(payload);
  return true;
}

static void send_error_response(const char *message)
{
  if (!message)
  {
    message = "Unknown error";
  }

  cJSON *err = cJSON_CreateObject();
  if (!err)
  {
    ESP_LOGE(TAG, "Failed to allocate JSON error response");
    return;
  }

  cJSON_AddStringToObject(err, "status", "error");
  cJSON_AddStringToObject(err, "message", message);
  send_ws_json(err);
  cJSON_Delete(err);
}

/* ======================== WIFI ======================== */

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
  if (!s_wifi_event_group)
    return;

  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START)
  {
    ESP_LOGI(TAG, "Wi-Fi STA start; connecting...");
    esp_wifi_connect();
  }
  else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED)
  {
    ESP_LOGW(TAG, "Wi-Fi disconnected; retrying...");
    xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    esp_wifi_connect();
  }
  else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
  {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
    ESP_LOGI(TAG, "Wi-Fi connected, IP: " IPSTR, IP2STR(&event->ip_info.ip));
    xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
  }
}

static esp_err_t wifi_init_sta(void)
{
  s_wifi_event_group = xEventGroupCreate();
  if (!s_wifi_event_group)
  {
    ESP_LOGE(TAG, "Failed to create Wi-Fi event group");
    return ESP_ERR_NO_MEM;
  }

  esp_err_t err = esp_netif_init();
  if (err != ESP_OK)
  {
    ESP_LOGE(TAG, "esp_netif_init failed: %s", esp_err_to_name(err));
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }

  err = esp_event_loop_create_default();
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE)
  {
    ESP_LOGE(TAG, "esp_event_loop_create_default failed: %s",
             esp_err_to_name(err));
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }
  else if (err == ESP_ERR_INVALID_STATE)
  {
    ESP_LOGW(TAG, "Event loop already created; continuing");
  }

  if (!esp_netif_create_default_wifi_sta())
  {
    ESP_LOGE(TAG, "Failed to create default Wi-Fi STA");
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return ESP_FAIL;
  }

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  err = esp_wifi_init(&cfg);
  if (err != ESP_OK)
  {
    ESP_LOGE(TAG, "esp_wifi_init failed: %s", esp_err_to_name(err));
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }
  err = esp_wifi_set_mode(WIFI_MODE_STA);
  if (err != ESP_OK)
  {
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
                  WIFI_AUTH_OPEN, /* Accept any auth mode (hotspot compatible) */
          },
  };
  err = esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
  if (err != ESP_OK)
  {
    ESP_LOGE(TAG, "esp_wifi_set_config failed: %s", esp_err_to_name(err));
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }

  bool wifi_handler_registered = false;
  bool ip_handler_registered = false;

  err = esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                   &wifi_event_handler, NULL);
  if (err != ESP_OK)
  {
    ESP_LOGE(TAG, "Failed to register Wi-Fi event handler: %s",
             esp_err_to_name(err));
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }
  wifi_handler_registered = true;

  err = esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                   &wifi_event_handler, NULL);
  if (err != ESP_OK)
  {
    ESP_LOGE(TAG, "Failed to register IP event handler: %s",
             esp_err_to_name(err));
    if (wifi_handler_registered)
    {
      esp_event_handler_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                   &wifi_event_handler);
    }
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }
  ip_handler_registered = true;

  err = esp_wifi_start();
  if (err != ESP_OK)
  {
    ESP_LOGE(TAG, "esp_wifi_start failed: %s", esp_err_to_name(err));
    if (wifi_handler_registered)
    {
      esp_event_handler_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                   &wifi_event_handler);
    }
    if (ip_handler_registered)
    {
      esp_event_handler_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                   &wifi_event_handler);
    }
    vEventGroupDelete(s_wifi_event_group);
    s_wifi_event_group = NULL;
    return err;
  }
  ESP_LOGI(TAG, "Connecting to Wi-Fi %s", WIFI_SSID);

  EventBits_t bits =
      xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE,
                          pdFALSE, pdMS_TO_TICKS(10000));

  if ((bits & WIFI_CONNECTED_BIT) == 0)
  {
    ESP_LOGE(TAG, "Wi-Fi connection timeout");
    if (wifi_handler_registered)
    {
      esp_event_handler_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                   &wifi_event_handler);
    }
    if (ip_handler_registered)
    {
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

/* ======================== CAMERA ======================== */

static esp_err_t camera_init(void)
{
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
      .frame_size = FRAMESIZE_QQVGA,
      .jpeg_quality = 20,
      .fb_count = 2,
  };

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK)
  {
    ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
    return err;
  }
  else
    ESP_LOGI(TAG, "Camera OK (QQVGA JPEG)");

  return ESP_OK;
}

/* ======================== WEBSOCKET EVENTS ======================== */

static void handle_lock_command(cJSON *root)
{
  const cJSON *action = cJSON_GetObjectItemCaseSensitive(root, "action");
  if (!action || !cJSON_IsString(action))
    return;

  if (strcmp(action->valuestring, "lock_grant") == 0)
  {
    const cJSON *user = cJSON_GetObjectItemCaseSensitive(root, "user");
    if (user && cJSON_IsString(user))
      ESP_LOGI(TAG, "ACCESS GRANTED: %s", user->valuestring);
    else
      ESP_LOGI(TAG, "ACCESS GRANTED");

    lock_open();

    /* Acknowledge */
    cJSON *ack = cJSON_CreateObject();
    if (ack)
    {
      cJSON_AddStringToObject(ack, "status", "ok");
      cJSON_AddStringToObject(ack, "lock", "opened");
      send_ws_json(ack);
      cJSON_Delete(ack);
    }
  }
  else if (strcmp(action->valuestring, "lock_deny") == 0)
  {
    ESP_LOGW(TAG, "ACCESS DENIED");

    /* Blink LED to indicate denial */
    for (int i = 0; i < 3; i++)
    {
      gpio_set_level(LED_GPIO, 0);
      vTaskDelay(pdMS_TO_TICKS(100));
      gpio_set_level(LED_GPIO, 1);
      vTaskDelay(pdMS_TO_TICKS(100));
    }
  }
}

static void handle_camera_settings(cJSON *root)
{
  sensor_t *s = esp_camera_sensor_get();
  if (!s)
  {
    send_error_response("Camera sensor unavailable");
    return;
  }

  bool updated = false;

  const cJSON *brightness =
      cJSON_GetObjectItemCaseSensitive(root, "brightness");
  if (brightness && cJSON_IsNumber(brightness))
  {
    s->set_brightness(s, brightness->valueint);
    ESP_LOGI(TAG, "Set brightness to %d", s->status.brightness);
    updated = true;
  }

  const cJSON *contrast = cJSON_GetObjectItemCaseSensitive(root, "contrast");
  if (contrast && cJSON_IsNumber(contrast))
  {
    s->set_contrast(s, contrast->valueint);
    ESP_LOGI(TAG, "Set contrast to %d", s->status.contrast);
    updated = true;
  }

  const cJSON *saturation =
      cJSON_GetObjectItemCaseSensitive(root, "saturation");
  if (saturation && cJSON_IsNumber(saturation))
  {
    s->set_saturation(s, saturation->valueint);
    ESP_LOGI(TAG, "Set saturation to %d", s->status.saturation);
    updated = true;
  }

  const cJSON *quality = cJSON_GetObjectItemCaseSensitive(root, "quality");
  if (quality && cJSON_IsNumber(quality))
  {
    s->set_quality(s, quality->valueint);
    ESP_LOGI(TAG, "Set quality to %d", s->status.quality);
    updated = true;
  }

  if (updated)
  {
    cJSON *resp = cJSON_CreateObject();
    if (resp)
    {
      cJSON_AddStringToObject(resp, "status", "ok");
      cJSON_AddStringToObject(resp, "message", "Camera settings updated");
      send_ws_json(resp);
      cJSON_Delete(resp);
    }
  }
}

static void on_ws_event(void *arg, esp_event_base_t base, int32_t eid,
                        void *data)
{
  esp_websocket_event_data_t *event = (esp_websocket_event_data_t *)data;

  switch (eid)
  {
  case WEBSOCKET_EVENT_CONNECTED:
    ESP_LOGI(TAG, "WebSocket connected");
    s_ws_connected = true;
    s_ws_started = true;
    break;

  case WEBSOCKET_EVENT_DISCONNECTED:
    ESP_LOGW(TAG, "WebSocket disconnected");
    s_ws_connected = false;
    break;

  case WEBSOCKET_EVENT_ERROR:
    s_ws_connected = false;
    if (event)
    {
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 2, 0)
      ESP_LOGE(TAG, "WebSocket error, type=%d, tls=0x%x",
               event->error_handle.error_type,
               event->error_handle.esp_tls_last_esp_err);
#else
      ESP_LOGE(TAG, "WebSocket error, type=%d, tls=0x%x, errno=%d",
               event->error_handle.error_type,
               event->error_handle.esp_tls_last_esp_err,
               event->error_handle.last_errno);
#endif
    }
    break;

  case WEBSOCKET_EVENT_DATA:
    if (!event)
      return;
    if (event->op_code == WS_TRANSPORT_OPCODES_BINARY)
      return;

    /* Parse incoming JSON command */
    char *json = strndup(event->data_ptr, event->data_len);
    if (!json)
      return;

    cJSON *root = cJSON_Parse(json);
    if (!root)
    {
      free(json);
      return;
    }

    /* Route by "action" field (lock commands) or camera settings */
    const cJSON *action = cJSON_GetObjectItemCaseSensitive(root, "action");
    if (action)
      handle_lock_command(root);
    else
      handle_camera_settings(root);

    cJSON_Delete(root);
    free(json);
    break;

  default:
    break;
  }
}

/* ======================== MAIN ======================== */

void app_main(void)
{
  ESP_LOGI(TAG, "=== Smart Lock ESP32-CAM ===");

  /* NVS */
  esp_err_t nvs_ret = nvs_flash_init();
  if (nvs_ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      nvs_ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
  {
    ESP_ERROR_CHECK(nvs_flash_erase());
    nvs_ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(nvs_ret);

  /* Wi-Fi */
  ESP_ERROR_CHECK(wifi_init_sta());

  /* Lock GPIO */
  lock_gpio_init();

  /* Camera */
  ESP_ERROR_CHECK(camera_init());

  /* WebSocket client */
  esp_websocket_client_config_t ws_cfg = {
      .uri = SERVER_URI,
      .buffer_size = WS_BUFFER_SIZE,
      .task_stack = 6144,
      .keep_alive_enable = true,
      .keep_alive_idle = 5,
      .keep_alive_interval = 5,
      .keep_alive_count = 3,
      .ping_interval_sec = 10,
      .pingpong_timeout_sec = 30,
      .disable_auto_reconnect = false,
      .enable_close_reconnect = true,
      .reconnect_timeout_ms = WS_RECONNECT_DELAY_MS,
      .network_timeout_ms = 10000,
  };

  ws = esp_websocket_client_init(&ws_cfg);
  if (!ws)
  {
    ESP_LOGE(TAG, "WebSocket client init failed");
    return;
  }

  esp_websocket_register_events(ws, WEBSOCKET_EVENT_ANY, on_ws_event, NULL);
  ESP_ERROR_CHECK(esp_websocket_client_start(ws));
  s_ws_started = true;
  ESP_LOGI(TAG, "WebSocket started: %s", SERVER_URI);

  /* Main loop: capture frames and stream to edge */
  int consecutive_failures = 0;
  int consecutive_null_frames = 0;
  uint32_t last_disconnect_log = 0;
  uint32_t last_stats_log = 0;
  uint32_t frames_sent = 0;
  uint32_t frames_capture_fail = 0;

  while (true)
  {
    /* Auto-close lock after timeout */
    lock_check_timeout();

    /* Wait if Wi-Fi is down */
    if (!s_wifi_event_group ||
        (xEventGroupGetBits(s_wifi_event_group) & WIFI_CONNECTED_BIT) == 0)
    {
      vTaskDelay(pdMS_TO_TICKS(250));
      continue;
    }

    /* Wait if WebSocket is disconnected */
    if (!ws || !esp_websocket_client_is_connected(ws))
    {
      uint32_t now_ms = esp_log_timestamp();
      if (now_ms - last_disconnect_log >= 2000)
      {
        last_disconnect_log = now_ms;
        ESP_LOGW(TAG, "WebSocket disconnected; waiting for reconnect...");
      }
      consecutive_failures = 0;
      vTaskDelay(pdMS_TO_TICKS(WS_RECONNECT_DELAY_MS));
      continue;
    }

    /* Capture and send frame */
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb)
    {
      consecutive_null_frames = 0;

      if (fb->format != PIXFORMAT_JPEG || fb->len == 0)
      {
        ESP_LOGW(TAG, "Invalid frame (format=%d, len=%u)",
                 fb->format, (unsigned)fb->len);
        esp_camera_fb_return(fb);
        vTaskDelay(pdMS_TO_TICKS(FRAME_INTERVAL_MS));
        continue;
      }

      int sent = esp_websocket_client_send_bin(
          ws, (const char *)fb->buf, fb->len,
          pdMS_TO_TICKS(WS_SEND_TIMEOUT_MS));

      if (sent < 0)
      {
        consecutive_failures++;
        ESP_LOGW(TAG, "Frame send failed (%d/%d)",
                 consecutive_failures, MAX_SEND_FAILURES);

        if (consecutive_failures >= MAX_SEND_FAILURES)
        {
          ESP_LOGE(TAG, "Too many failures, forcing reconnect");
          s_ws_connected = false;
          (void)esp_websocket_client_close(ws, pdMS_TO_TICKS(500));
          consecutive_failures = 0;
        }
      }
      else
      {
        consecutive_failures = 0;
        frames_sent++;
      }

      esp_camera_fb_return(fb);
    }
    else
    {
      frames_capture_fail++;
      consecutive_null_frames++;

      if (consecutive_null_frames >= MAX_CAMERA_NULL_FRAMES)
      {
        ESP_LOGW(TAG, "Camera returned NULL frame %d times, reinitializing camera",
                 consecutive_null_frames);
        esp_camera_deinit();
        if (camera_init() == ESP_OK)
        {
          ESP_LOGI(TAG, "Camera reinitialized successfully");
        }
        consecutive_null_frames = 0;
      }
    }

    uint32_t now_ms = esp_log_timestamp();
    if (now_ms - last_stats_log >= 5000)
    {
      last_stats_log = now_ms;
      ESP_LOGI(TAG, "Stream stats: sent=%u, capture_fail=%u", frames_sent,
               frames_capture_fail);
      frames_sent = 0;
      frames_capture_fail = 0;
    }

    vTaskDelay(pdMS_TO_TICKS(FRAME_INTERVAL_MS));
  }
}
