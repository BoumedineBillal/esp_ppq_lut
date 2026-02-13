
import torch
import torch.nn as nn
from enum import Enum
from typing import Any
from esp_ppq.utils.round import ppq_tensor_round
from esp_ppq.core import RoundingPolicy, TargetPlatform
from esp_ppq.executor.base import OPERATION_FORWARD_TABLE
from esp_ppq.executor.op.torch.default import DEFAULT_BACKEND_TABLE
from esp_ppq.parser.espdl.espdl_typedef import ACTIVATION_OP_SET, PASSIVE_LAYOUT_OP_SET

class SimulationMode(Enum):
    SIMULATION = 1
    IDEAL_MATH = 2

def set_simulation_mode(mode: SimulationMode):
    """Convenience helper to set the global emulator mode."""
    GlobalMode.set(mode)

class GlobalMode:
    """Manages the current mode of the ESP-DL emulator."""
    _current_mode = SimulationMode.SIMULATION

    @classmethod
    def set(cls, mode: SimulationMode):
        cls._current_mode = mode

    @classmethod
    def get(cls) -> SimulationMode:
        return cls._current_mode

class HardwareEmulator(torch.autograd.Function):
    """
    Generalized Bit-Exact Hardware Emulator for ESP-DL LUT Operations.
    Implements: output = x + trunc((len * (y - x)) / step)
    """
    @staticmethod
    def forward(ctx, input_tensor, math_fn, op_context, in_scale, out_scale, step, rounding):
        # Store context for backward pass
        ctx.math_fn = math_fn
        ctx.op_context = op_context
        ctx.save_for_backward(input_tensor)

        # 1. Quantize Input to Signed INT16 range
        # Hardware logic: input is already quantized to INT16 by previous layer
        # Ensure scale is broadcastable [1, C, 1, 1] for [N, C, H, W]
        if isinstance(in_scale, torch.Tensor) and in_scale.ndim > 0:
            in_scale = in_scale.view(1, -1, 1, 1) if input_tensor.ndim == 4 else in_scale
            
        input_int = ppq_tensor_round(input_tensor / in_scale, rounding)
        # Standard INT16 signed range: -32768 to 32767
        input_signed = torch.clamp(input_int, -32768, 32767).to(torch.int32)

        # 2. Linear Indexing
        # idx_shifted maps [-32768, 32767] to [0, 65535]
        idx_shifted = input_signed + 32768
        base_idx = idx_shifted // step
        remainder = idx_shifted % step

        # 3. Pivot Calculation (Ideal Domain)
        # We find the real-world values of the two nearest pivots
        x_int = (base_idx * step) - 32768
        y_int = x_int + step
        
        x_real = x_int * in_scale
        y_real = y_int * in_scale

        # 4. Call Ideal Math Function for Pivots
        # math_fn is the pure mathematical activation from OPERATION_FORWARD_TABLE
        x_ideal = math_fn(op_context, [x_real])
        y_ideal = math_fn(op_context, [y_real])

        # 5. Quantize Pivots for Truncated Interpolation
        # Hardware logic: table values are stored as INT16
        if isinstance(out_scale, torch.Tensor) and out_scale.ndim > 0:
            out_scale = out_scale.view(1, -1, 1, 1) if x_ideal.ndim == 4 else out_scale
            
        x_quant = ppq_tensor_round(x_ideal / out_scale, rounding).clamp(-32768, 32767)
        y_quant = ppq_tensor_round(y_ideal / out_scale, rounding).clamp(-32768, 32767)

        # 6. ESP-DL Fixed-Point Linear Interpolation
        # formula: x + trunc(len * (y - x) / step)
        delta_y = y_quant - x_quant
        interpolation = torch.trunc((remainder * delta_y) / step)
        output_quant = x_quant + interpolation

        # 7. Rescale to Float for Pipeline Consistency
        return output_quant.clamp(-32768, 32767) * out_scale

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        math_fn = ctx.math_fn
        op_context = ctx.op_context

        # STE (Straight-Through Estimator):
        # We calculate the gradient of the IDEAL function for training stability
        with torch.enable_grad():
            x = input_tensor.detach().requires_grad_(True)
            # Pull math from the registered forward tables
            y = math_fn(op_context, [x])
            grad = torch.autograd.grad(y.sum(), x)[0]
        
        return grad_output * grad, None, None, None, None, None, None

def lut_forward_provider(op, values, ctx=None, **kwargs):
    """
    The 'Universal LUT Execution Engine'.
    Switches between Ideal Math (for export) and Bit-Exact Simulation (for everything else).
    """
    input_tensor = values[0]
    
    # 1. Strict Attribute Verification Loop
    mandatory_attrs = ['original_op_type', 'int16_lut_step']
    for attr in mandatory_attrs:
        if attr not in op.attributes:
            print(f"\033[91m[CRITICAL ERROR] Mandatory attribute '{attr}' missing for Operation: {op.name}\033[0m")
            raise AttributeError(f"Required attribute '{attr}' is missing. Check fusion pipeline.")
            
    original_type = op.attributes['original_op_type']
    step          = op.attributes['int16_lut_step']
    
    # 2. Find the Mathematical Ideal (The 'Truth')
    # The 'Mathematical Truth' lives in the DEFAULT_BACKEND_TABLE
    if original_type not in DEFAULT_BACKEND_TABLE:
        print(f"\033[91m[CRITICAL ERROR] Math implementation for '{original_type}' not found in DEFAULT_BACKEND_TABLE.\033[0m")
        raise KeyError(f"Mathematical ground truth for {original_type} is missing.")
    ideal_math_fn = DEFAULT_BACKEND_TABLE[original_type]
            
    if ideal_math_fn is None:
        raise KeyError(f"Could not find ideal math function for {original_type} in DEFAULT_BACKEND_TABLE.")

    # 3. Path A: IDEAL MODE (For Table Generation during Export)
    if GlobalMode.get() == SimulationMode.IDEAL_MATH:
        return ideal_math_fn(op, values)

    # 4. Path B: SIMULATION MODE (Default: For Validation, PTQ, and STE)
    # We extract scales and rounding from the operation's configuration
    # Note: These attributes are populated by PPQ during quantization
    in_scale = op.input_quant_config[0].scale
    out_scale = op.output_quant_config[0].scale
    rounding = op.input_quant_config[0].rounding

    return HardwareEmulator.apply(
        input_tensor, ideal_math_fn, op, 
        in_scale, out_scale, step, rounding
    )

def register_lut_op_handler(verbose=False):
    """
    Registers the LUT operation handler globally.
    This enables hardware-aware simulation for LUT operations.
    """
    ACTIVATION_OP_SET.add("LUT")
    PASSIVE_LAYOUT_OP_SET.add("LUT")
    
    # Inject our provider into ALL platforms in the global forward table
    for platform in OPERATION_FORWARD_TABLE:
        OPERATION_FORWARD_TABLE[platform]['LUT'] = lut_forward_provider
    
    if verbose:
        print("[ESPDL Emulator] LUT Operation Handler Registered Globally.")
