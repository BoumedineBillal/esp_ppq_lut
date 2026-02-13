
from typing import List
from esp_ppq.IR import BaseGraph, Operation
from esp_ppq.quantization.optim import QuantizationOptimizationPass
from esp_ppq.core import TargetPlatform
from esp_ppq.executor.torch import OPERATION_FORWARD_TABLE

class ESPDL_LUTFusionPass(QuantizationOptimizationPass):
    """
    Graph Re-writing pass for ESP-DL Deployment.
    Converts activation operations (Swish, Sigmoid) into 'LUT' operations
    to trigger the Hardware-Accelerated Fast Path in the ESP-DL Runtime.
    """
    def __init__(self, target_ops: List[str] = ['Swish', 'Sigmoid', 'Tanh'], 
                 verify: bool = False, deep_verify: bool = False, plot: bool = False, 
                 output_dir: str = "outputs", lut_step: int = 32):
        super().__init__(name='ESPDL LUT Fusion Pass')
        self.target_ops = target_ops
        self.verify = verify
        self.deep_verify = deep_verify
        self.plot = plot
        self.output_dir = output_dir
        self.lut_step = lut_step
        self.lut_count = 0

    def verify_compatibility(self, op: Operation) -> bool:
        """
        The 'Safety Layer': Verifies if the operation is suitable for LUT conversion.
        """
        # Placeholder for curvature-based safety checks.
        # Currently returns True as per architectural discussion.
        return True

    def _self_audit(self, op: Operation, index: int, executor=None, calib_data=None):
        """
        Performs bit-exact parity check and optional exhaustive sweep.
        """
        import os
        import torch
        from .patches import AddLUTPattern
        from .emulator import set_simulation_mode, SimulationMode
        from .utils import run_numerical_verification, generate_comparison_plot, to_c_header, update_verification_manifest
        
        pattern = AddLUTPattern(int16_step=self.lut_step)
        verify_dir = os.path.join(self.output_dir, "lut_verification")
        
        metadata = {
            "layer_name": op.name,
            "original_type": op.attributes.get('original_op_type', 'Unknown'),
            "parity_plot": f"lut_{index}_parity.png"
        }

        # 1. Pivot Parity Check (INT16 Domain)
        # ---------------------------------------------------------------------
        set_simulation_mode(SimulationMode.IDEAL_MATH)
        table_ideal = pattern.calculate_lut(op, None, max=32767, min=-32768, step=self.lut_step)
        
        set_simulation_mode(SimulationMode.SIMULATION)
        table_sim = pattern.calculate_lut(op, None, max=32767, min=-32768, step=self.lut_step)
        
        run_numerical_verification(
            table_sim.flatten()[:2048], 
            table_ideal.flatten()[:2048], 
            title=f"Parity: {op.name}", 
            is_table=True,
            output_dir=verify_dir,
            filename=f"lut_{index}_parity"
        )

        # 2. Deep Verification: Exhaustive Sweep + C Header
        # ---------------------------------------------------------------------
        if self.deep_verify and executor is not None:
            print(f"[ESPDL Pass] Running Exhaustive Sweep for layer: {op.name}")
            scale = op.input_quant_config[0].scale.item()
            
            # Exhaustive INT16 sweep + OOB
            full_sweep = torch.arange(-32768, 32768, dtype=torch.float)
            oob = torch.tensor([-40000.0, -32769.0, 32768.0, 40000.0])
            test_data = torch.cat([oob[:2], full_sweep, oob[2:]]).view(1, 1, 1, -1) * scale
            
            # Simulation
            y_sim = executor(test_data)[0]
            
            # Ideal Math (Dynamic Lookup)
            from esp_ppq.executor.op.torch.default import DEFAULT_BACKEND_TABLE
            math_fn = DEFAULT_BACKEND_TABLE[op.attributes['original_op_type']]
            y_ideal = math_fn(op, [test_data])
            
            # Calibration Data Plotting
            y_calib_sim = None
            if calib_data is not None:
                y_calib_sim = executor(calib_data)[0]

            # Generate Artifacts
            sweep_filename = f"lut_{index}_sweep.png"
            header_filename = f"lut_{index}_test.h"
            
            generate_comparison_plot(
                test_data, y_sim, y_ideal, 
                x_calib=calib_data, y_calib=y_calib_sim,
                output_path=os.path.join(verify_dir, sweep_filename)
            )
            to_c_header(test_data, f"input_{index}", os.path.join(verify_dir, header_filename))
            to_c_header(y_sim, f"output_{index}", os.path.join(verify_dir, header_filename))
            
            metadata["sweep_plot"] = sweep_filename
            metadata["c_header"] = header_filename

        # Update JSON Manifest
        update_verification_manifest(verify_dir, index, metadata)

    def optimize(self, graph: BaseGraph, **kwargs):
        """
        Traverses the graph and performs the 'Topology Swap'.
        """
        import os
        self.lut_count = 0
        executor = kwargs.get('executor', None)
        dataloader = kwargs.get('dataloader', None)
        calib_data = None
        if dataloader is not None:
            # Get a sample batch for plotting
            calib_data = dataloader[0] if isinstance(dataloader, list) else next(iter(dataloader))
        
        # Clear/Init verification directory if enabled
        if self.verify or self.deep_verify:
            verify_dir = os.path.join(self.output_dir, "lut_verification")
            os.makedirs(verify_dir, exist_ok=True)
            manifest_file = os.path.join(verify_dir, "mapping.json")
            if os.path.exists(manifest_file):
                os.remove(manifest_file)

        for op in graph.operations.values():
            if op.type in self.target_ops:
                # 1. Safety Check
                if not self.verify_compatibility(op):
                    continue

                # 2. Store Metadata (The 'Shadow Attributes')
                op.attributes['original_op_type'] = op.type
                op.attributes['int16_lut_step'] = self.lut_step
                
                # 3. Rename the Operation Type
                op.type = 'LUT'
                print(f"[ESPDL Pass] Fused {op.attributes['original_op_type']} -> LUT for operation: {op.name}")

                # 4. Self-Audit (Optional)
                self.lut_count += 1
                if self.verify or self.deep_verify:
                    self._self_audit(op, self.lut_count, executor=executor, calib_data=calib_data)

    @property
    def is_post_quantization_pass(self) -> bool:
        return True
