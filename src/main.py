
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import esp_ppq.lib as PFL
from esp_ppq.api import load_onnx_graph
from esp_ppq.core import TargetPlatform, QuantizationVisibility, RoundingPolicy
from esp_ppq.executor import TorchExecutor
from esp_ppq.executor.base import OPERATION_FORWARD_TABLE
from esp_ppq.parser.espdl.espdl_typedef import ACTIVATION_OP_SET, PASSIVE_LAYOUT_OP_SET
from esp_ppq.quantization.optim import (
    QuantizeSimplifyPass, QuantizeFusionPass, ParameterQuantizePass,
    RuntimeCalibrationPass, PassiveParameterQuantizePass, QuantAlignmentPass
)

# LUT Extension Import
import esp_ppq_lut as esp_lut

# =================================================================================================
# 1. ARCHITECTURAL INITIALIZATION
# =================================================================================================

# Global Configuration
LUT_STEP = 32
TARGET_PLATFORM = TargetPlatform.ESPDL_INT16
VERBOSE = False # Set to True for detailed deployment logs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the LUT Extension Package
# This handles all global registrations (Op Handlers, Exporters)
esp_lut.initialize(step=LUT_STEP, verbose=VERBOSE)

# =================================================================================================
# 2. MODEL & PIPELINE DEFINITION
# =================================================================================================

class SwishModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.conv.weight.data.fill_(1.0) # Identity weight for pure activation test

    def forward(self, x):
        x = self.conv(x)
        x = x * torch.sigmoid(x)
        return x

def run_deployment():
    # 1. Setup Data
    calibration_data = torch.randn(1, 1, 32, 32)
    onnx_file = os.path.join(OUTPUT_DIR, "swish_test.onnx")
    torch.onnx.export(SwishModule(), calibration_data, onnx_file, opset_version=14)

    # 2. Load and Basic Quantization
    graph = load_onnx_graph(onnx_import_file=onnx_file)
    quantizer = PFL.Quantizer(platform=TARGET_PLATFORM, graph=graph)
    
    # Manually dispatch everything to INT16
    for op in graph.operations.values():
        quantizer.quantize_operation(op_name=op.name, platform=TARGET_PLATFORM)

    # 3. Standard Quantization Pipeline (The 'Math' Stage)
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

    # 4. Deployment Pipeline (The 'Hardware Mapping' Stage)
    # This pass now handles its own bit-exact parity audit!
    deployment_pipeline = PFL.Pipeline([
        esp_lut.EspdlLUTFusionPass(
            target_ops=['Swish'], # Supported: 'Swish', 'Sigmoid', 'Tanh'
            verify=True, 
            plot=True, 
            output_dir=OUTPUT_DIR,
            lut_step=LUT_STEP,
            verbose=VERBOSE
        )
    ])
    deployment_pipeline.optimize(graph=graph, dataloader=[calibration_data], executor=executor)


    # 5. Standalone Deep Verification (Optional Debug Step)
    # NOTE: This step is designed for isolated activation verification.
    esp_lut.run_deep_verification(
        graph=graph, 
        executor=executor, 
        dataloader=[calibration_data], 
        output_dir=OUTPUT_DIR,
        verbose=VERBOSE
    )


    # 6. Export
    espdl_path = os.path.join(OUTPUT_DIR, "swish_test.espdl")
    PFL.Exporter(platform=TARGET_PLATFORM).export(
        espdl_path, graph=graph, int16_lut_step=LUT_STEP
    )
    if VERBOSE:
        print(f"\n[DEPLOYMENT] Model Exported: {espdl_path}")

if __name__ == "__main__":
    run_deployment()
