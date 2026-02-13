
import torch
from esp_ppq.IR import Variable
from esp_ppq.IR.quantize import QuantableOperation
from esp_ppq.executor.base import OPERATION_FORWARD_TABLE
from esp_ppq.parser.espdl.espdl_typedef import ExporterPatternInfo
from esp_ppq.parser.espdl.export_patterns import AddLUTPattern
from esp_ppq import PPQLinearQuant_toInt
from esp_ppq.utils.round import ppq_tensor_round

def patched_get_scale(self, var: Variable, info: ExporterPatternInfo) -> torch.Tensor:
    """
    Library Fix: Prevents the exporter from defaulting to 2^exponent scales
    which can create mismatches with the real TQC scales used in simulation.
    """
    # 1. First Priority: Check if info has pre-computed exponents (ESPDL standard)
    if info is not None and hasattr(info, 'get_var_exponents'):
        exponent = info.get_var_exponents(var.name)
        if exponent:
            if isinstance(exponent, list):
                return 2 ** exponent[0]
            else:
                return 2 ** exponent
            
    # Fallback to the real TQC scale if no metadata exponent is present
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
    """
    Library Fix: ESP-DL interpolation requires a fencepost point (2049 points for step-32).
    Also ensures the math used for table generation is identical to the simulation logic.
    """
    platform_dispatching_table = OPERATION_FORWARD_TABLE[op.platform]
    # Library Fix: Ensure the operation type exists in the dispatching table
    if op.type not in platform_dispatching_table:
        print(f"\033[91m[CRITICAL ERROR] Operation type '{op.type}' not found in dispatching table for {op.platform.name}.\033[0m")
        raise KeyError(f"Missing forward function for {op.type}")
        
    operation_forward_func = platform_dispatching_table[op.type]
    
    # 2049 points fix: min to max + step
    input = torch.arange(min, max + step, step=step, dtype=torch.float)
    
    # Use the patched scale retrieval
    scale = self.get_scale(op.inputs[0], info)
    input = input * scale
    inputs = [input]

    if len(op.inputs) > 1:
        for op_input in op.inputs[1:]:
            inputs.append(op_input.value * self.get_scale(op_input, info))
            
    output = operation_forward_func(op, inputs)
    device = op.output_quant_config[0].scale.device
    
    # Quantize to INT16 pivots
    lut = PPQLinearQuant_toInt(output.to(device), op.output_quant_config[0])
    return lut

def patch_esp_ppq_library():
    """Applies essential library fixes for ESP-DL LUT generation."""
    AddLUTPattern.get_scale = patched_get_scale
    AddLUTPattern.calculate_lut = patched_calculate_lut
    print("[ESP-PPQ-LUT] Applied library monkey patches for AddLUTPattern parity.")
