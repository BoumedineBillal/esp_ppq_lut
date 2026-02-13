from .emulator import register_lut_op_handler, set_simulation_mode, SimulationMode
from . import emulator
from .exporter import register_espdl_exporter
from .passes import ESPDL_LUTFusionPass
from .utils import run_numerical_verification, to_c_header, generate_comparison_plot
from .patches import patch_esp_ppq_library

def initialize(step=32):
    """
    The entry point for the ESP-PPQ LUT extension.
    Initializes hardware-aware execution and deployment plugins.
    """
    # Apply essential library fixes first
    patch_esp_ppq_library()
    
    # Register handlers and exporters
    register_lut_op_handler()
    register_espdl_exporter()
    print("[ESP-PPQ-LUT] Applied library monkey patches for AddLUTPattern parity.")
    print(f"[ESP-PPQ-LUT] Extension Initialized (Default Step={step})")
