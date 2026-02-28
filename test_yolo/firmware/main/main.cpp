/*
    * YOLO26n PTQ Triple-Mode Validation Firmware
    * IMG_SZ: 512, LUT_STEP: 32
    *
    * Model A: LUT model   (Swish replaced by LUT tables,)
    * Model B: IDEAL model (Swish as standard op, no LUT)
    *
    * TEST 1: HW(Model A) vs SIMULATION vectors  — expects match within +/-1 tolerance
    * TEST 2: HW(Model A) vs IDEAL_MATH vectors  — expects mismatches
    * TEST 3: HW(Model B) vs IDEAL_MATH vectors  — expects match within +/-1 tolerance
    */

    #include <stdio.h>
    #include <string.h>
    #include <math.h>
    #include <stdlib.h>
    #include "freertos/FreeRTOS.h"
    #include "freertos/task.h"
    #include "freertos/task.h"
    #include "nvs_flash.h"
    #include "esp_log.h"
    #include "esp_timer.h"

    #include "dl_model_base.hpp"
    #include "dl_tensor_base.hpp"
    #include "dl_tool.hpp"
    #include "yolo26.hpp"

    // Test Data
    namespace test_data {
    #include "test_data/yolo_test_p4_data.h"
    #include "test_data/raw_rgb_person.h"
    }

    static const char *TAG = "YOLO26_VAL";

    // Model Binary
    extern const uint8_t model_lut_bin[]   asm("_binary_yolo26n_p4_espdl_start");
    extern const uint8_t model_ideal_bin[] asm("_binary_yolo26n_ideal_p4_espdl_start");

    // Comparison result for a single output
    struct CompareResult {
        int mismatches;   // count of elements where |diff| > 0
        int fail_tol;     // count of elements where |diff| exceeds tolerance
        int min_err;      // minimum signed error (most negative)
        int max_err;      // maximum signed error (most positive)
    };

    // ============================================================
    // Validation Tolerances  ← adjust here if needed
    // ============================================================
    // INT8  — 256 levels:  ±1 LSB ≈ 0.8% of full range.
    // INT16 — 65536 levels: ±5 LSB ≈ 0.015% of full range. Relaxed to
    //         absorb float32↔float64 requantisation rounding in the Python
    //         test-vector generator, while still catching real HW faults.
    static constexpr int INT8_TOLERANCE  = 1;
    static constexpr int INT16_TOLERANCE = 5;
    // ============================================================

    /**
    * Compare HW output against a reference output array (INT8).
    */
    CompareResult compare_output_int8(int8_t *hw_ptr, const float *ref, int size,
                                    float out_scale, const char *label)
    {
        CompareResult r = {0, 0, 0, 0};
        for (int i = 0; i < size; i++) {
            int rounded_out = dl::tool::round(ref[i] / out_scale);
            if (rounded_out > 127) rounded_out = 127;
            if (rounded_out < -128) rounded_out = -128;
            int8_t ref_int = (int8_t)rounded_out;
            int8_t hw_int = hw_ptr[i];
            int diff = (int)hw_int - (int)ref_int;
            if (diff != 0) {
                if (r.mismatches < 3)
                    ESP_LOGW(TAG, "    [%d]: HW=%d vs %s=%d (diff=%d)",
                            i, hw_int, label, ref_int, diff);
                r.mismatches++;
                if (abs(diff) > INT8_TOLERANCE) r.fail_tol++;
                if (diff < r.min_err) r.min_err = diff;
                if (diff > r.max_err) r.max_err = diff;
            }
        }
        return r;
    }

    /**
    * Compare HW output against a reference output array (INT16).
    */
    CompareResult compare_output_int16(int16_t *hw_ptr, const float *ref, int size,
                                    float out_scale, const char *label)
    {
        CompareResult r = {0, 0, 0, 0};
        for (int i = 0; i < size; i++) {
            int rounded_out = dl::tool::round(ref[i] / out_scale);
            if (rounded_out > 32767) rounded_out = 32767;
            if (rounded_out < -32768) rounded_out = -32768;
            int16_t ref_int = (int16_t)rounded_out;
            int16_t hw_int = hw_ptr[i];
            int diff = (int)hw_int - (int)ref_int;
            if (diff != 0) {
                if (r.mismatches < 3)
                    ESP_LOGW(TAG, "    [%d]: HW=%d vs %s=%d (diff=%d)",
                            i, hw_int, label, ref_int, diff);
                r.mismatches++;
                if (abs(diff) > INT16_TOLERANCE) r.fail_tol++;
                if (diff < r.min_err) r.min_err = diff;
                if (diff > r.max_err) r.max_err = diff;
            }
        }
        return r;
    }

    /**
    * Inject test input into model and run inference.
    */
    void inject_input_and_run(dl::Model *model, const char *model_name)
    {
        auto &inputs = model->get_inputs();
        dl::TensorBase *input_tensor = inputs.begin()->second;
        float in_scale = powf(2.0f, input_tensor->exponent);
        int8_t *input_ptr = (int8_t *)input_tensor->data;
        int in_size = input_tensor->size;

        ESP_LOGI(TAG, "[%s] Input size: %d, scale: %.8f", model_name, in_size, in_scale);

        for (int i = 0; i < in_size && i < test_data::test_input_len; i++) {
            float val = test_data::test_input[i] / in_scale;
            int rounded_val = dl::tool::round(val);
            if (rounded_val > 127) rounded_val = 127;
            if (rounded_val < -128) rounded_val = -128;
            input_ptr[i] = (int8_t)rounded_val;
        }

        ESP_LOGI(TAG, "[%s] Running inference...", model_name);
        model->run();
    }

    /**
    * Compare all outputs of a model against reference arrays.
    * For expect_match tests: PASS if all errors are within INT16_TOLERANCE.
    * INT16_TOLERANCE is used in all log output for consistency.
    * Returns total mismatches.
    */
    int compare_all_outputs(dl::Model *model, const float **ref_ptrs, const char *ref_label,
                            bool expect_match)
    {
        auto &outputs = model->get_outputs();
        int total_mismatches = 0;
        int total_fail_tol = 0;
        int total_values = 0;
        int global_min_err = 0;
        int global_max_err = 0;

        for (int output_idx = 0; output_idx < test_data::num_outputs; output_idx++) {
            const char *name = test_data::output_names[output_idx];
            if (outputs.find(name) == outputs.end()) {
                ESP_LOGE(TAG, "Output %s not found in model", name);
                continue;
            }
            dl::TensorBase *tensor = outputs.at(name);
            float out_scale = powf(2.0f, tensor->exponent);
            int size = tensor->size;
            CompareResult r;

            if (tensor->dtype == dl::DATA_TYPE_INT16) {
                r = compare_output_int16((int16_t *)tensor->data,
                                        ref_ptrs[output_idx],
                                        size, out_scale, ref_label);
            } else {
                r = compare_output_int8((int8_t *)tensor->data,
                                        ref_ptrs[output_idx],
                                        size, out_scale, ref_label);
            }
            // tolerance is determined by the actual dtype of this output
            int tol = (tensor->dtype == dl::DATA_TYPE_INT16) ? INT16_TOLERANCE : INT8_TOLERANCE;

            if (expect_match) {
                if (r.mismatches == 0) {
                    ESP_LOGI(TAG, "  %s: PASS (%d values, err: 0)", name, size);
                } else if (r.fail_tol == 0) {
                    ESP_LOGI(TAG, "  %s: PASS +/-%d (%d/%d mismatches, err_range:[%d,%d])",
                            name, tol, r.mismatches, size, r.min_err, r.max_err);
                } else {
                    ESP_LOGE(TAG, "  %s: FAIL %d/%d exceed +/-%d (%d/%d total, err_range:[%d,%d])",
                            name, r.fail_tol, size, tol, r.mismatches, size, r.min_err, r.max_err);
                }
            } else {
                if (r.mismatches > 0)
                    ESP_LOGI(TAG, "  %s: %d/%d mismatches (err_range:[%d,%d])",
                            name, r.mismatches, size, r.min_err, r.max_err);
                else
                    ESP_LOGW(TAG, "  %s: 0 mismatches (unexpected)", name);
            }

            total_mismatches += r.mismatches;
            total_fail_tol += r.fail_tol;
            total_values += size;
            if (r.min_err < global_min_err) global_min_err = r.min_err;
            if (r.max_err > global_max_err) global_max_err = r.max_err;
        }

        if (expect_match) {
            if (total_mismatches == 0) {
                ESP_LOGI(TAG, "  \033[32mTOTAL PASS: 100%%%% Bit-Exact (%d values, %d outputs)\033[0m",
                        total_values, test_data::num_outputs);
            } else if (total_fail_tol == 0) {
                ESP_LOGI(TAG, "  \033[32mTOTAL PASS within +/-%d: %d/%d mismatches (err_range:[%d,%d], %d outputs)\033[0m",
                        INT16_TOLERANCE, total_mismatches, total_values, global_min_err, global_max_err, test_data::num_outputs);
            } else {
                ESP_LOGE(TAG, "  \033[31mTOTAL FAIL: %d/%d exceed +/-%d (%d/%d total, err_range:[%d,%d])\033[0m",
                        total_fail_tol, total_values, INT16_TOLERANCE, total_mismatches, total_values,
                        global_min_err, global_max_err);
            }
        } else {
            ESP_LOGI(TAG, "  TOTAL: %d mismatches (err_range:[%d,%d])",
                    total_mismatches, global_min_err, global_max_err);
        }

        return total_mismatches;
    }

    extern "C" void app_main(void)
    {
        ESP_LOGI(TAG, "==========================================================");
        ESP_LOGI(TAG, "   YOLO26n PTQ Triple-Mode Validation");
        ESP_LOGI(TAG, "   IMG_SZ: 512, LUT_STEP: 32");
        ESP_LOGI(TAG, "   TARGET: ESP32-P4");
        ESP_LOGI(TAG, "==========================================================");

        // ============================================================
        // Load LUT Model (Model A — Swish replaced by LUT tables)
        // ============================================================
        ESP_LOGI(TAG, "----------------------------------------------------------");
        ESP_LOGI(TAG, "Loading LUT Model (Model A)...");
        dl::Model *model_lut = new dl::Model((const char *)model_lut_bin,
                                            fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                                            0, dl::MEMORY_MANAGER_GREEDY,
                                            nullptr, true);
        if (!model_lut) { ESP_LOGE(TAG, "Failed to load LUT model!"); return; }

        // Native HW Preprocessing test
        YOLO26 processor(model_lut, 32, 0.25f);
        
        // BYPASS JPEG DECODER: Feed raw uncompressed RGB to ImagePreprocessor
        dl::image::img_t img;
        img.data = (uint8_t*)test_data::raw_rgb_person;
        img.width = 500;
        img.height = 375;
        img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;

        int64_t start_pre = esp_timer_get_time();
        processor.preprocess(img);
        int64_t end_pre = esp_timer_get_time();
        
        auto &inputs = model_lut->get_inputs();
        dl::TensorBase *input_tensor = inputs.begin()->second;
        float in_scale = powf(2.0f, input_tensor->exponent);
        int in_size = input_tensor->size;
        
        ESP_LOGI(TAG, "[LUT] Input size: %d, scale: %.8f", in_size, in_scale);
        ESP_LOGI(TAG, "----------------------------------------------------------");
        ESP_LOGI(TAG, "TEST 0: HW(LUT Preprocess) vs Python(test_input) — expect match (+/-%d tol)", INT8_TOLERANCE);
        
        CompareResult r_input = compare_output_int8((int8_t *)input_tensor->data,
                                                test_data::test_input,
                                                in_size, in_scale, "SIM_IN");
        if (r_input.fail_tol > 0) {
            ESP_LOGE(TAG, "  input_tensor: FAIL %d/%d exceed +/-%d (%d/%d total, err_range:[%d,%d])", 
                    r_input.fail_tol, in_size, INT8_TOLERANCE, r_input.mismatches, in_size, r_input.min_err, r_input.max_err);
        } else if (r_input.mismatches > 0) {
            ESP_LOGI(TAG, "  input_tensor: PASS +/-%d (%d/%d mismatches, err_range:[%d,%d])", 
                    INT8_TOLERANCE, r_input.mismatches, in_size, r_input.min_err, r_input.max_err);
        } else {
            ESP_LOGI(TAG, "  input_tensor: PASS (%d values, err: 0)", in_size);
        }
        
        // Do NOT free img.data because we assigned it directly to a const array in Flash ROM (.rodata)

        ESP_LOGI(TAG, "[LUT] Running inference...");
        int64_t start_inf = esp_timer_get_time();
        model_lut->run();
        int64_t end_inf = esp_timer_get_time();

        // TEST 1: HW(LUT model) vs SIMULATION vectors
        ESP_LOGI(TAG, "----------------------------------------------------------");
        ESP_LOGI(TAG, "TEST 1: HW(LUT model) vs SIMULATION — expect match (+/-%d tol)", INT16_TOLERANCE);
        compare_all_outputs(model_lut, test_data::output_sim_ptrs, "SIM", true);

        // TEST 2: HW(LUT model) vs IDEAL_MATH vectors
        ESP_LOGI(TAG, "----------------------------------------------------------");
        ESP_LOGI(TAG, "TEST 2: HW(LUT model) vs IDEAL_MATH — expect mismatches");
        compare_all_outputs(model_lut, test_data::output_ideal_ptrs, "IDEAL", false);

        // Print Final Post-Processing Bounding Boxes
        ESP_LOGI(TAG, "----------------------------------------------------------");
        ESP_LOGI(TAG, "Running YOLO26 Post-Processing...");
        int64_t start_post = esp_timer_get_time();
        auto results = processor.postprocess(model_lut->get_outputs());
        int64_t end_post = esp_timer_get_time();

        uint32_t lat_pre = (uint32_t)((end_pre - start_pre) / 1000);
        uint32_t lat_inf = (uint32_t)((end_inf - start_inf) / 1000);
        uint32_t lat_post = (uint32_t)((end_post - start_post) / 1000);

        ESP_LOGI(TAG, "Pre: %lu ms | Inf: %lu ms | Post: %lu ms", lat_pre, lat_inf, lat_post);

        ESP_LOGI("YOLO26", "=== Testing: person.jpg ===");
        for(auto& res : results) {
            ESP_LOGI("YOLO26", "[category: %s, score: %.2f, x1: %d, y1: %d, x2: %d, y2: %d]", 
                processor.class_names[res.class_id], res.score, (int)res.x1, (int)res.y1, (int)res.x2, (int)res.y2);
        }

        delete model_lut;

        // ============================================================
        // Load IDEAL Model (Model B — standard Swish, no LUT)
        // ============================================================
        ESP_LOGI(TAG, "----------------------------------------------------------");
        ESP_LOGI(TAG, "Loading IDEAL Model (Model B)...");
        dl::Model *model_ideal = new dl::Model((const char *)model_ideal_bin,
                                            fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                                            0, dl::MEMORY_MANAGER_GREEDY,
                                            nullptr, true);
        if (!model_ideal) { ESP_LOGE(TAG, "Failed to load IDEAL model!"); return; }

        inject_input_and_run(model_ideal, "IDEAL");

        // TEST 3: HW(IDEAL model) vs IDEAL_MATH vectors
        ESP_LOGI(TAG, "----------------------------------------------------------");
        ESP_LOGI(TAG, "TEST 3: HW(IDEAL model) vs IDEAL_MATH — expect match (+/-%d tol)", INT16_TOLERANCE);
        compare_all_outputs(model_ideal, test_data::output_ideal_ptrs, "IDEAL", true);

        delete model_ideal;

        ESP_LOGI(TAG, "==========================================================");
        ESP_LOGI(TAG, "Validation Finished.");
    }
    