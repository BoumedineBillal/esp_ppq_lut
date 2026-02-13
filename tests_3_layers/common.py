import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import esp_ppq.lib as PFL
from esp_ppq.api import load_onnx_graph, get_target_platform
from esp_ppq.core import TargetPlatform, QuantizationVisibility
from esp_ppq.executor import TorchExecutor
from esp_ppq.quantization.optim import (
    QuantizeSimplifyPass, QuantizeFusionPass, ParameterQuantizePass,
    RuntimeCalibrationPass, PassiveParameterQuantizePass, QuantAlignmentPass
)

# Add parent directory to path to import esp_ppq_lut
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PARENT_DIR, "src")
sys.path.append(SRC_DIR)

import esp_ppq_lut as esp_lut

# Global Configuration
LUT_STEP = 32
TARGET_PLATFORM = TargetPlatform.ESPDL_INT16
FIRMWARE_DIR = os.path.join(SCRIPT_DIR, "firmware")
TEST_DATA_DIR = os.path.join(FIRMWARE_DIR, "main", "test_data")
MODELS_DIR = os.path.join(FIRMWARE_DIR, "main", "models")
os.makedirs(TEST_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Model Definitions
# ------------------------------------------------------------------------------

class SwishModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.conv.weight.data.fill_(1.0)
    def forward(self, x):
        x = self.conv(x)
        return x * torch.sigmoid(x)

class SigmoidModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.conv.weight.data.fill_(1.0)
    def forward(self, x):
        return torch.sigmoid(self.conv(x))

class TanhModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.conv.weight.data.fill_(1.0)
    def forward(self, x):
        return torch.tanh(self.conv(x))

# ------------------------------------------------------------------------------
# Standard Pipeline Execution
# ------------------------------------------------------------------------------

def run_layer_test_for_target(model_class, layer_name, activation_op, target="esp32p4"):
    platform = get_target_platform(target, 16)
    target_suffix = "p4" if "p4" in target else "s3"
    
    print(f"\n{'='*60}")
    print(f"Running Validation for: {layer_name} on {target.upper()}")
    print(f"{'='*60}")
    
    # Initialize Package
    esp_lut.initialize(step=LUT_STEP, verbose=False)
    
    # Setup Output
    output_dir = os.path.join(SCRIPT_DIR, "outputs", target_suffix, layer_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Export ONNX
    calibration_data = torch.randn(1, 1, 32, 32)
    onnx_file = os.path.join(output_dir, f"{layer_name}.onnx")
    torch.onnx.export(model_class(), calibration_data, onnx_file, opset_version=14)
    
    # 2. Quantization Pipeline
    graph = load_onnx_graph(onnx_import_file=onnx_file)
    quantizer = PFL.Quantizer(platform=platform, graph=graph)
    for op in graph.operations.values():
        quantizer.quantize_operation(op_name=op.name, platform=platform)
        
    quant_pipeline = PFL.Pipeline([
        QuantizeSimplifyPass(),
        QuantizeFusionPass(activation_type=quantizer.activation_fusion_types),
        ParameterQuantizePass(),
        RuntimeCalibrationPass(method='minmax'),
        PassiveParameterQuantizePass(clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE),
        QuantAlignmentPass(elementwise_alignment='Align to Output'),
    ])
    
    executor = TorchExecutor(graph=graph, device='cpu')
    executor.tracing_operation_meta(inputs=calibration_data)
    quant_pipeline.optimize(graph=graph, dataloader=[calibration_data], executor=executor)
    
    # 3. Fusion & Verification Pipeline
    deployment_pipeline = PFL.Pipeline([
        esp_lut.EspdlLUTFusionPass(
            target_ops=[activation_op], 
            verify=True, 
            plot=True, 
            output_dir=output_dir,
            lut_step=LUT_STEP,
            verbose=False # Keep silent by default
        )
    ])
    deployment_pipeline.optimize(graph=graph, dataloader=[calibration_data], executor=executor)
    
    # 4. Deep Verification & C Header Generation
    # This step generates the artifacts for firmware validation
    esp_lut.run_deep_verification(
        graph=graph, 
        executor=executor, 
        dataloader=[calibration_data], 
        output_dir=output_dir,
        verbose=False
    )
    
    # Copy header to firmware folder for easy access
    import shutil
    # The verifier generates headers named lut_1_test.h, lut_2_test.h etc based on loop index
    # For a single layer model, it is always lut_1_test.h
    src_header = os.path.join(output_dir, "lut_verification", "lut_1_test.h")
    dst_header = os.path.join(TEST_DATA_DIR, f"{layer_name}_{target_suffix}_data.h")
    
    if os.path.exists(src_header):
        shutil.copy(src_header, dst_header)
        print(f"[Firmware] Data header copied to: {dst_header}")
    else:
        print(f"\033[91m[Error] Could not find generated header at {src_header}\033[0m")

    # 5. Export .espdl
    espdl_path = os.path.join(output_dir, f"{layer_name}.espdl")
    PFL.Exporter(platform=platform).export(
        espdl_path, graph=graph, int16_lut_step=LUT_STEP
    )
    
    # Copy .espdl to firmware folder
    dst_espdl = os.path.join(MODELS_DIR, f"{layer_name}_{target_suffix}.espdl")
    if os.path.exists(espdl_path):
        shutil.copy(espdl_path, dst_espdl)
        print(f"[Firmware] Model binary copied to: {dst_espdl}")

    print(f"[Success] Validation complete for {layer_name} ({target.upper()})")

def run_layer_test(model_class, layer_name, activation_op):
    """Backwards compatibility / Convenience wrapper for both targets."""
    for target in ["esp32p4", "esp32s3"]:
        run_layer_test_for_target(model_class, layer_name, activation_op, target=target)
