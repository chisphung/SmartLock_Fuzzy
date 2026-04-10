idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES esp32-camera esp_websocket_client esp_wifi esp_netif esp_event cjson nvs_flash driver
    PRIV_REQUIRES spi_flash esp_wifi cjson nvs_flash
)
