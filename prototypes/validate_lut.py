
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.onnx
import os
import matplotlib.pyplot as plt
import esp_ppq.lib as PFL
from esp_ppq.api import load_onnx_graph
from esp_ppq.core import TargetPlatform, QuantizationVisibility, RoundingPolicy
from esp_ppq.utils.round import ppq_tensor_round
from esp_ppq.executor.base import OPERATION_FORWARD_TABLE
from esp_ppq.executor import TorchExecutor
from esp_ppq.parser.espdl.espdl_typedef import ACTIVATION_OP_SET, PASSIVE_LAYOUT_OP_SET, ExporterPatternInfo
from esp_ppq.quantization.optim import (
    QuantizeSimplifyPass, QuantizeFusionPass, ParameterQuantizePass,
    RuntimeCalibrationPass, PassiveParameterQuantizePass, QuantAlignmentPass
)
from esp_ppq.parser.espdl.export_patterns import AddLUTPattern
from esp_ppq import PPQLinearQuant_toInt
from esp_ppq.IR import Variable
from esp_ppq.IR.quantize import QuantableOperation

# =================================================================================================
# 0. SETUP & DIRECTORIES
# =================================================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Project Root: {SCRIPT_DIR}")
print(f"Output Dir:   {OUTPUT_DIR}")

# =================================================================================================
# 1. MONKEY PATCH: Fix ESP-PPQ Fencepost Error (2048 -> 2049 points)
# =================================================================================================
def patched_get_scale(self, var: Variable, info: ExporterPatternInfo) -> torch.Tensor:
    exponent = info.get_var_exponents(var.name)
    if exponent:
        if isinstance(exponent, list):
            return 2 ** exponent[0]
        else:
            return 2**exponent
            
    # Fallback for manual verification: Use the scale from the variable's quant config
    if hasattr(var, 'dest_ops') and len(var.dest_ops) > 0:
        op = var.dest_ops[0]
        if isinstance(op, QuantableOperation):
            try:
                idx = op.inputs.index(var)
                return op.input_quant_config[idx].scale
            except (ValueError, IndexError):
                pass
                
    return torch.tensor(1.0, device=var.value.device if var.value is not None else 'cpu')

def patched_calculate_lut(self, op: QuantableOperation, info: ExporterPatternInfo, max: int, min: int, step: int = 1) -> torch.Tensor:
    platform_dispatching_table = OPERATION_FORWARD_TABLE[op.platform]
    operation_forward_func = platform_dispatching_table[op.type]
    
    # THE UNIVERSAL FIX: Use 'max + step' instead of 'max + 1'
    input = torch.arange(min, max + step, step=step, dtype=torch.float)
    input = input * self.get_scale(op.inputs[0], info)
    inputs = [input]

    if len(op.inputs) > 1:
        for op_input in op.inputs[1:]:
            inputs.append(op_input.value * self.get_scale(op_input, info))
            
    output = operation_forward_func(op, inputs)
    device = op.output_quant_config[0].scale.device
    lut = PPQLinearQuant_toInt(output.to(device), op.output_quant_config[0])
    
    print(f"input scale: {op.input_quant_config[0].scale.item()}")
    print(f"output scale: {op.output_quant_config[0].scale.item()}")
    print(f"scale input : self.get_scale(op.inputs[0], info)  {self.get_scale(op.inputs[0], info)}")
    
    return lut

AddLUTPattern.get_scale = patched_get_scale
AddLUTPattern.calculate_lut = patched_calculate_lut
print("PATCH APPLIED: AddLUTPattern.calculate_lut optimized for interpolation (2049 points).")

# =================================================================================================
# 2. Hardware Simulation Logic (The "Digital Twin")
# =================================================================================================
USE_BIT_EXACT_SIMULATION = True 

class HardwareLUT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, in_scale, out_scale, step, rounding):
        ctx.save_for_backward(input_tensor)
        input_int = ppq_tensor_round(input_tensor / in_scale, rounding)
        input_signed = torch.clamp(input_int, -32768, 32767)

        idx_shifted = input_signed + 32768
        base_idx = idx_shifted // step
        remainder = idx_shifted % step
        from esp_ppq.executor.op.torch.default import DEFAULT_BACKEND_TABLE
        math_fn = DEFAULT_BACKEND_TABLE['Swish']

        x_int = (base_idx * step) - 32768
        x_real = x_int * in_scale
        x_ideal = math_fn(None, [x_real]) 

        y_int = x_int + step
        y_real = y_int * in_scale
        y_ideal = math_fn(None, [y_real])

        x = ppq_tensor_round(x_ideal / out_scale, rounding).clamp(-32768, 32767)
        y = ppq_tensor_round(y_ideal / out_scale, rounding).clamp(-32768, 32767)

        output_quant = x + torch.trunc((remainder * (y - x)) / step)
        return output_quant.clamp(-32768, 32767) * out_scale

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        # Use SiLU (Swish) derivative for STE
        sig = torch.sigmoid(input_tensor)
        swish_grad = sig * (1 + input_tensor * (1 - sig))
        return grad_output * swish_grad, None, None, None, None

def lut_forward_clean(op, values, ctx=None, **kwargs):
    input_tensor = values[0]
    LUT_STEP = 32 
    
    in_scale = 1.0
    out_scale = 1.0
    
    if hasattr(op, 'config') and len(op.config.input_quantization_config) > 0:
        s = op.config.input_quantization_config[0].scale
        in_scale = s if isinstance(s, float) else s.to(input_tensor.device).view(1, -1, 1, 1)

    if hasattr(op, 'config') and len(op.config.output_quantization_config) > 0:
        s = op.config.output_quantization_config[0].scale
        out_scale = s if isinstance(s, float) else s.to(input_tensor.device).view(1, -1, 1, 1)

    rounding = RoundingPolicy.ROUND_HALF_EVEN
    if hasattr(op, 'config') and len(op.config.input_quantization_config) > 0:
        rounding = op.config.input_quantization_config[0].rounding

    # CRITICAL FIX: During Table Generation (calculate_lut), we MUST use the real function.
    # The Exporter usually calls this through get_scale/calculate_lut which sets context.
    # We check if the input tensor has exactly 2049 elements (typical for step-32 INT16 table).
    # Or more generally: Table generation happens in high-precision float domain for pivots.
    if USE_BIT_EXACT_SIMULATION:
        print("Using HardwareLUT")
        return HardwareLUT.apply(input_tensor, in_scale, out_scale, LUT_STEP, rounding)
    else:
        # Table Generation / Ideal Math path
        from esp_ppq.executor.op.torch.default import DEFAULT_BACKEND_TABLE
        math_fn = DEFAULT_BACKEND_TABLE['Swish']
        print("Using Ideal Math")
        return math_fn(op, [input_tensor])

ACTIVATION_OP_SET.add("LUT")
PASSIVE_LAYOUT_OP_SET.add("LUT")
for platform in OPERATION_FORWARD_TABLE:
    OPERATION_FORWARD_TABLE[platform]['LUT'] = lut_forward_clean

# =================================================================================================
# 3. Model & Data Generation
# =================================================================================================
class SingleSwish(nn.Module):
    def __init__(self):
        super(SingleSwish, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, 1, 1))

    def forward(self, x):
        x = F.conv2d(x, self.weight, bias=None)
        x = x * torch.sigmoid(x)
        return x

model = SingleSwish()
calibration_data = torch.randn(1, 1, 32, 32)
test_data = None # Will be generated after calibration to use exact scale

# =================================================================================================
# 4. Quantization Pipeline
# =================================================================================================
target_platform = TargetPlatform.ESPDL_INT16

# 1. Export to ONNX
onnx_file = os.path.join(OUTPUT_DIR, "swish_lut_test.onnx")
torch.onnx.export(
    model, calibration_data, onnx_file, opset_version=14,
    input_names=['input'], output_names=['output']
)

# 2. Load and Quantize
graph = load_onnx_graph(onnx_import_file=onnx_file)
quantizer = PFL.Quantizer(platform=target_platform, graph=graph)
dispatching_table = PFL.Dispatcher(graph=graph, method="conservative").dispatch(quantizer.quant_operation_types)

for op in graph.operations.values():
    dispatching_table[op.name] = target_platform
    quantizer.quantize_operation(op_name=op.name, platform=dispatching_table[op.name])

# 3. Calibration Pipeline
pipeline = PFL.Pipeline([
    QuantizeSimplifyPass(),
    QuantizeFusionPass(activation_type=quantizer.activation_fusion_types),
    ParameterQuantizePass(),
    RuntimeCalibrationPass(method='minmax'),
    PassiveParameterQuantizePass(clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE),
    QuantAlignmentPass(elementwise_alignment='Align to Output'),
])

executor = TorchExecutor(graph=graph, device='cpu')
executor.tracing_operation_meta(inputs=calibration_data)
pipeline.optimize(graph=graph, dataloader=[calibration_data], executor=executor, verbose=True)

# Patch Swish to LUT
for op in graph.operations.values():
    if op.type == 'Swish':
        op.type = 'LUT'

# =================================================================================================
# 5. Verification Tools
# =================================================================================================
def run_numerical_verification(y_sim, y_ideal, title="Verification"):
    y_sim = y_sim.flatten()
    y_ideal = y_ideal.flatten()
    
    mse = F.mse_loss(y_sim, y_ideal).item()
    max_err = torch.max(torch.abs(y_sim - y_ideal)).item()
    matches_mask = torch.isclose(y_sim, y_ideal, atol=1e-7)
    matches = matches_mask.float().mean().item() * 100
    
    print(f"\n--- {title} ---")
    print(f"MSE:        {mse:.10f}")
    print(f"Max Error:  {max_err:.10f}")
    print(f"Match %:    {matches:.2f}%")

    s_np = y_sim.detach().cpu().numpy()
    i_np = y_ideal.detach().cpu().numpy()
    indices = np.arange(len(s_np))
    diff_mask = ~matches_mask.cpu().numpy()
    
    # Identify mismatch segments
    segments = []
    padding = 10
    start = None
    for idx in range(len(diff_mask)):
        if diff_mask[idx] and start is None: start = idx
        elif not diff_mask[idx] and start is not None:
            segments.append((max(0, start - padding), min(len(diff_mask)-1, idx + padding)))
            start = None
    if start is not None: segments.append((max(0, start - padding), len(diff_mask)-1))
    
    merged = []
    if segments:
        merged = [segments[0]]
        for cur in segments[1:]:
            prev = merged[-1]
            if cur[0] <= prev[1] + 5: merged[-1] = (prev[0], max(prev[1], cur[1]))
            else: merged.append(cur)
    
    # Console Detail
    if np.any(diff_mask):
        exact_segments = []
        start = None
        for idx in range(len(diff_mask)):
            if diff_mask[idx] and start is None: start = idx
            elif not diff_mask[idx] and start is not None:
                exact_segments.append((start, idx - 1))
                start = None
        if start is not None: exact_segments.append((start, len(diff_mask) - 1))

        print(f"\n[DETAIL] Mismatched Segments Found: {len(exact_segments)}")
        for i, (s, e) in enumerate(exact_segments):
            mismatch_indices = [idx for idx in range(s, e + 1) if diff_mask[idx]]
            sim_vals = [f"{s_np[idx]:.4f}" for idx in mismatch_indices]
            ideal_vals = [f"{i_np[idx]:.4f}" for idx in mismatch_indices]
            print(f"  Seg {i+1} [{s}:{e}] | Idx: {mismatch_indices} | Sim: {sim_vals} | Ideal: {ideal_vals}")
    else:
        print("\n[DETAIL] No bit-exact mismatches found.")

    # Plotting
    plot_segments = merged[:4]
    num_subplots = 1 + len(plot_segments)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 4 + 3 * len(plot_segments)))
    if num_subplots == 1: axes = [axes]
    
    ax_main = axes[0]
    ax_main.plot(indices, s_np, label='Bit-Exact (Sim)', color='red', linewidth=1.5)
    ax_main.plot(indices, i_np, label='Ideal (F32)', linestyle='--', color='blue', linewidth=1)
    if np.any(diff_mask):
        ax_main.fill_between(indices, ax_main.get_ylim()[0], ax_main.get_ylim()[1], where=diff_mask, color='red', alpha=0.1)
    ax_main.set_title(f"Comparison: {title}")
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.3)

    for i, (seg_start, seg_end) in enumerate(plot_segments):
        ax = axes[i+1]
        ax.plot(indices[seg_start:seg_end], s_np[seg_start:seg_end], marker='o', markersize=4, label='Sim', color='red')
        ax.plot(indices[seg_start:seg_end], i_np[seg_start:seg_end], marker='x', markersize=4, label='Ideal', color='blue', linestyle='--')
        ax.set_title(f"Mismatch Segment {i+1} (Indices {seg_start}:{seg_end})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.tight_layout()
    safe_title = title.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plt.savefig(os.path.join(OUTPUT_DIR, f"verify_{safe_title}.png"))
    plt.close() 

def verify_lut_comparison(graph):
    global USE_BIT_EXACT_SIMULATION
    print("\n[VERIFICATION] Comparing Manual LUT Table Generation...")
    lut_ops = [op for op in graph.operations.values() if op.type in ['LUT', 'Swish']]
    if not lut_ops:
        print("\033[91m[CRITICAL ERROR] No LUT/Swish operation found in graph.\033[0m")
        raise RuntimeError("Refusing to proceed without target operation.")
    lut_op = lut_ops[0]
    if not lut_op: return

    info = ExporterPatternInfo()
    pattern = AddLUTPattern(int16_step=32)
    
    # 1. Generate the Table using Math (This is what the Exporter writes to .espdl)
    USE_BIT_EXACT_SIMULATION = False
    lut_ideal = pattern.calculate_lut(lut_op, info, max=32767, min=-32768, step=32)
    
    # 2. Extract pivot values from the Simulation for comparison
    # We want to see if HardwareLUT(pivot) == Math(pivot)
    USE_BIT_EXACT_SIMULATION = True
    lut_be = pattern.calculate_lut(lut_op, info, max=32767, min=-32768, step=32)
    
    run_numerical_verification(lut_be.flatten()[0:2048].float(), lut_ideal.flatten()[0:2048].float(), title="LUT Pivot Parity (Math vs HW-Sim)")
    print("[VERIFICATION] Table comparison complete.")

# =================================================================================================
# 6. EXECUTION
# =================================================================================================

verify_lut_comparison(graph)

# Find the LUT operation to extract its calibrated scale
lut_ops = [op for op in graph.operations.values() if op.type in ['LUT', 'Swish']]
if not lut_ops:
    print("\033[91m[CRITICAL ERROR] No LUT/Swish operation found in graph for sweep.\033[0m")
    raise RuntimeError("Mandatory operation missing for exhaustive sweep.")
lut_op = lut_ops[0]
scale = lut_op.input_quant_config[0].scale.item()

# Exhaustive INT16 sweep (65,536 points) + Out-of-Bounds Stress Test
full_sweep = torch.arange(-32768, 32768, dtype=torch.float)
oob_low    = torch.tensor([-40000.0, -32769.0])
oob_high   = torch.tensor([32768.0, 40000.0])

# Combine and SCALE to match the real-world dynamic range of the model
raw_data = torch.cat([oob_low, full_sweep, oob_high])
# Pad to multiple of 1024 (32*32) for firmware looping alignment
padding_needed = (1024 - (len(raw_data) % 1024)) % 1024
if padding_needed > 0:
    raw_data = torch.cat([raw_data, torch.zeros(padding_needed)])

test_data = raw_data.view(1, 1, 1, -1) * scale

# Run Simulation on both Test and Calibration Data
y_sim = executor(test_data)[0]
y_calib_sim = executor(calibration_data)[0]
print(f"Simulation Complete. Output Shape: {y_sim.shape}")

# Export Model
USE_BIT_EXACT_SIMULATION = False
espdl_path = os.path.join(OUTPUT_DIR, "swish_lut_test.espdl")
PFL.Exporter(platform=target_platform).export(espdl_path, graph=graph, int16_lut_step=32)
print(f"Model Encoded: {espdl_path}")

# Export C Header
def to_c_array(tensor, name):
    data = tensor.flatten().detach().cpu().numpy()
    formatted_data = []
    for i in range(0, len(data), 128):
        chunk = data[i : i + 128]
        line = ", ".join([f"{x:.6f}f" for x in chunk])
        formatted_data.append(line)
    
    c_str = ",\n    ".join(formatted_data)
    return f"const float {name}[] = {{\n    {c_str}\n}};\nconst int {name}_len = {len(data)};\n"

header_path = os.path.join(OUTPUT_DIR, "test_data.h")
with open(header_path, "w") as f:
    f.write("#pragma once\n// Auto-generated\n")
    f.write(to_c_array(test_data, "input_data"))
    f.write(to_c_array(y_sim, "expected_output"))
print(f"Header Generated: {header_path}")

# Final Plot
from esp_ppq.executor.op.torch.default import DEFAULT_BACKEND_TABLE
if 'original_op_type' not in lut_op.attributes:
    print(f"\033[91m[CRITICAL ERROR] Mandatory attribute 'original_op_type' missing for LUT operation in validate_lut.py.\033[0m")
    raise AttributeError(f"Operation {lut_op.name} is missing 'original_op_type'. Ensure fusion pass is registered.")
    
original_type = lut_op.attributes['original_op_type']
if original_type not in DEFAULT_BACKEND_TABLE:
    print(f"\033[91m[CRITICAL ERROR] Math implementation for '{original_type}' not found in DEFAULT_BACKEND_TABLE.\033[0m")
    raise KeyError(f"Mathematical ground truth for {original_type} is missing.")
math_fn = DEFAULT_BACKEND_TABLE[original_type]

x_flat = test_data.flatten().detach().numpy()
y_sim_flat = y_sim.flatten().detach().numpy()
y_ideal_flat = math_fn(lut_op, [test_data]).flatten().detach().numpy()

x_calib_flat = calibration_data.flatten().detach().numpy()
y_calib_sim_flat = y_calib_sim.flatten().detach().numpy()

sort_idx = np.argsort(x_flat)
sort_idx_calib = np.argsort(x_calib_flat)

plt.figure(figsize=(12, 6))
plt.plot(x_flat[sort_idx], y_ideal_flat[sort_idx], label='Ideal Swish (F32)', color='blue', alpha=0.3, linestyle='--')
plt.plot(x_flat[sort_idx], y_sim_flat[sort_idx], label='Hardware LUT Sim (Test Data)', color='red', linewidth=1.5, alpha=0.8)
plt.scatter(x_calib_flat[sort_idx_calib], y_calib_sim_flat[sort_idx_calib], label='Hardware LUT Sim (Calib Data)', color='green', s=10, alpha=0.5)

# Calibration Data Range Markers
calib_min, calib_max = x_calib_flat.min(), x_calib_flat.max()
plt.axvline(x=calib_min, color='green', alpha=0.8, linewidth=0.5, label=f'Calib Min ({calib_min:.2f})')
plt.axvline(x=calib_max, color='green', alpha=0.8, linewidth=0.5, label=f'Calib Max ({calib_max:.2f})')

plt.title('Hardware-Exact Swish Simulation: Test vs Calibration Data')
plt.xlabel('Input Value')
plt.ylabel('Output Value')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "swish_comparison.png"), dpi=300)
plt.close()

print(f"Validation Finished. All outputs in: {OUTPUT_DIR}")
