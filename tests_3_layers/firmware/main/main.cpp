/*
 * Hardware-Exact LUT Validation App (3-Layer Suite)
 * * Objective: Confirm that the ESP32-P4 Hardware execution of INT16 LUTs
 * for Swish, Sigmoid, and Tanh is 100% BIT-EXACT to the Python Simulation.
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_log.h"

// ------------------------------------------------------------------
// ESP-DL Includes
// ------------------------------------------------------------------
#include "dl_model_base.hpp" 
#include "dl_tensor_base.hpp"
#include "dl_tool.hpp" // CRITICAL: For official dl::tool::round (P4 RNE behavior)

// ------------------------------------------------------------------
// Generated Test Data (Include target-specific headers)
// ------------------------------------------------------------------
#if defined(CONFIG_IDF_TARGET_ESP32P4)

#if __has_include("test_data/swish_test_p4_data.h")
namespace swish_data {
#include "test_data/swish_test_p4_data.h"
}
#define HAS_SWISH 1
#endif

#if __has_include("test_data/sigmoid_test_p4_data.h")
namespace sigmoid_data {
#include "test_data/sigmoid_test_p4_data.h"
}
#define HAS_SIGMOID 1
#endif

#if __has_include("test_data/tanh_test_p4_data.h")
namespace tanh_data {
#include "test_data/tanh_test_p4_data.h"
}
#define HAS_TANH 1
#endif

#elif defined(CONFIG_IDF_TARGET_ESP32S3)

#if __has_include("test_data/swish_test_s3_data.h")
namespace swish_data {
#include "test_data/swish_test_s3_data.h"
}
#define HAS_SWISH 1
#endif

#if __has_include("test_data/sigmoid_test_s3_data.h")
namespace sigmoid_data {
#include "test_data/sigmoid_test_s3_data.h"
}
#define HAS_SIGMOID 1
#endif

#if __has_include("test_data/tanh_test_s3_data.h")
namespace tanh_data {
#include "test_data/tanh_test_s3_data.h"
}
#define HAS_TANH 1
#endif

#endif // CONFIG_IDF_TARGET

static const char *TAG = "LUT_VALIDATION";

// Generic Validation Function
bool validate_layer(const char* name, 
                    const uint8_t* model_ptr, 
                    const float* input_data, 
                    const float* expected_output, 
                    int data_len) 
{
    ESP_LOGI(TAG, "----------------------------------------------------------");
    ESP_LOGI(TAG, "Testing Layer: %s", name);
    
    dl::Model *model = new dl::Model((const char *)model_ptr, 
                                     fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                                     0, dl::MEMORY_MANAGER_GREEDY,
                                     nullptr, true);
    
    if (!model) {
        ESP_LOGE(TAG, "Failed to load model for %s!", name);
        return false;
    }

    // 1. Setup Tensors
    dl::TensorBase *input_tensor = model->get_inputs().begin()->second;
    dl::TensorBase *output_tensor = model->get_outputs().begin()->second;
    
    float in_scale = powf(2.0f, input_tensor->exponent);
    float out_scale = powf(2.0f, output_tensor->exponent);
    int16_t *input_ptr = (int16_t *)input_tensor->data;
    int16_t *output_ptr = (int16_t *)output_tensor->data;

    int chunk_size = input_tensor->size;
    int num_chunks = data_len / chunk_size;
    int fail_count = 0;

    // 2. Process Data in Chunks
    for (int c = 0; c < num_chunks; c++) {
        // A. Inject Data
        for (int i = 0; i < chunk_size; i++) {
            int g_idx = c * chunk_size + i;
            float val = input_data[g_idx] / in_scale; 
            int rounded_val = dl::tool::round(val);
            
            // Standard Clamping (Safe for all toolchains)
            if (rounded_val > 32767) rounded_val = 32767;
            if (rounded_val < -32768) rounded_val = -32768;
            input_ptr[i] = (int16_t)rounded_val;
        }

        // B. Run Inference
        model->run();

        // C. Verify Bit-Exactness
        for (int i = 0; i < chunk_size; i++) {
            int g_idx = c * chunk_size + i;
            int16_t sim_int = (int16_t)roundf(expected_output[g_idx] / out_scale);
            if (output_ptr[i] != sim_int) {
                if (fail_count < 5) {
                    ESP_LOGE(TAG, "  Mismatch at %d: HW=%d vs Sim=%d", g_idx, output_ptr[i], sim_int);
                }
                fail_count++;
            }
        }
    }

    if (fail_count == 0) {
        ESP_LOGI(TAG, "  \033[32m✅ SUCCESS: %s matches 100%% (Bit-Exact)\033[0m", name);
    } else {
        ESP_LOGE(TAG, "  \033[31m❌ FAILURE: %s found %d mismatches\033[0m", name, fail_count);
    }

    delete model;
    return fail_count == 0;
}

// ------------------------------------------------------------------
// Model Binary Symbols (Auto-linked by target_add_aligned_binary_data)
// ------------------------------------------------------------------
#if defined(CONFIG_IDF_TARGET_ESP32P4)
    extern const uint8_t swish_model[]   asm("_binary_swish_test_p4_espdl_start");
    extern const uint8_t sigmoid_model[] asm("_binary_sigmoid_test_p4_espdl_start");
    extern const uint8_t tanh_model[]    asm("_binary_tanh_test_p4_espdl_start");
#else // ESP32S3
    extern const uint8_t swish_model[]   asm("_binary_swish_test_s3_espdl_start");
    extern const uint8_t sigmoid_model[] asm("_binary_sigmoid_test_s3_espdl_start");
    extern const uint8_t tanh_model[]    asm("_binary_tanh_test_s3_espdl_start");
#endif

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "==========================================================");
    ESP_LOGI(TAG, "   Starting Multi-Layer LUT Validation Loop");
    #if defined(CONFIG_IDF_TARGET_ESP32P4)
        ESP_LOGI(TAG, "   TARGET: ESP32-P4");
    #else
        ESP_LOGI(TAG, "   TARGET: ESP32-S3");
    #endif
    ESP_LOGI(TAG, "==========================================================");

    #ifdef HAS_SWISH
        validate_layer("Swish", swish_model, swish_data::input_1, swish_data::output_1, swish_data::input_1_len);
    #endif

    #ifdef HAS_SIGMOID
        validate_layer("Sigmoid", sigmoid_model, sigmoid_data::input_1, sigmoid_data::output_1, sigmoid_data::input_1_len);
    #endif

    #ifdef HAS_TANH
        validate_layer("Tanh", tanh_model, tanh_data::input_1, tanh_data::output_1, tanh_data::input_1_len);
    #endif

    ESP_LOGI(TAG, "==========================================================");
    ESP_LOGI(TAG, "Validation Suite Finished.");
}
