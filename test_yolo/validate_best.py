"""
YOLO26n QAT Validation Script (Standalone)
==========================================
Loads a previously saved Best_yolo26n.native graph and runs full COCO mAP validation.
"""

import torch
import numpy as np
import os
import sys
import types
import shutil
import cv2
from ultralytics.data.augment import LetterBox


if __name__ == '__main__':
    # ==========================================
    # 1. PATH SETUP
    # ==========================================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # validate_lut_exact_3/
    SRC_DIR = os.path.join(ROOT_DIR, "src")
    SCRIPTS_DIR = os.path.join(SCRIPT_DIR, "scripts")

    sys.path.insert(0, SRC_DIR)       # for esp_ppq_lut
    sys.path.insert(0, SCRIPTS_DIR)   # for export, dataset, utils, esp_ppq_patch

    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # CONFIGURATION
    # ==========================================
    PLATFORM = "p4"    # "p4" or "s3"
    LUT_STEP = 32      # INT16 LUT interpolation step
    IMG_SZ = 512       # Input image size


    # ==========================================
    # 2. IMPORTS & INITIALIZATION
    # ==========================================

    # -- Inject virtual config module for scripts/*.py compat --
    from esp_ppq.api import get_target_platform
    from ultralytics.data.utils import check_det_dataset

    class QATConfig:
        EPOCHS = 0  # MODIFIED: No training
        BATCH_SIZE = 20
        IMG_SZ = IMG_SZ
        DATA_FRACTION = 1.0
        SEED = 1234
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        OPTIMIZER_LR = 1e-6
        OPTIMIZER_MOMENTUM = 0.937
        OPTIMIZER_WEIGHT_DECAY = 5e-4
        DATA_YAML_FILE = "coco.yaml"
        DATA_FALLBACK_PATH = "coco2017/images/train2017"
        CALIB_MAX_IMAGES = 8192
        CALIB_VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        TARGET_PLATFORM = get_target_platform("esp32" + PLATFORM, 8)
        CALIB_STEPS = 64
        QUANT_CALIB_METHOD = "kl"
        QUANT_ALIGNMENT = "Align to Output"
        EXPORT_OPSET = 13
        EXPORT_DYNAMIC = False
        LOSS_DEFAULTS = {
            'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0,
            'label_smoothing': 0.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7,
            'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
            'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5,
            'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0,
        }
        BASE_DIR = SCRIPT_DIR
        MODEL_NAME = "yolo26n"
        PT_FILE = f"{MODEL_NAME}.pt"
        ONNX_FILE = f"{MODEL_NAME}_train.onnx"
        ONNX_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_train.onnx")
        ESPDL_OUTPUT_DIR = OUTPUT_DIR
        VAL_PLOT_MAX_BATCHES = 3
        VAL_BATCH_SIZE = 24


    if 'config' in sys.modules:
        sys.modules['config'].QATConfig = QATConfig
    else:
        config_module = types.ModuleType('config')
        config_module.QATConfig = QATConfig
        sys.modules['config'] = config_module

    # -- Import project dependencies --
    import esp_ppq.lib as PFL
    from esp_ppq.executor import TorchExecutor
    from esp_ppq.core import QuantizationVisibility, TargetPlatform
    from esp_ppq.api.interface import load_onnx_graph
    from esp_ppq.quantization.optim import (
        QuantizeSimplifyPass, QuantizeFusionPass, ParameterQuantizePass,
        RuntimeCalibrationPass, PassiveParameterQuantizePass, QuantAlignmentPass
    )
    from esp_ppq.IR import BaseGraph

    # -- Import scripts --
    from utils import seed_everything, register_mod_op, patch_v8_detection_loss, get_exclusive_ancestors
    from dataset import get_calibration_loader
    from export import apply_export_patches, ESP_YOLO
    from esp_ppq_patch import apply_esp_ppq_patches

    # ==========================================
    # MONKEY PATCH: Fix AddLUTPattern Step Size
    # ==========================================
    try:
        from esp_ppq.parser.espdl.export_patterns import (
            AddLUTPattern, Operation, BaseGraph, QuantableOperation,
            EspQuantType, ExporterPatternInfo
        )
        def patched_export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
            quant_type = op.attributes.get("quant_type", None)
            if quant_type == None or quant_type == EspQuantType.F32 or not isinstance(op, QuantableOperation):
                return op
            info = ExporterPatternInfo()
            if self.check_op(op):
                lut = None
                if quant_type == EspQuantType.S8:
                    lut = self.calculate_lut(op, info, 127, -128, 1)
                elif quant_type == EspQuantType.S16:
                    current_step = op.attributes.get("int16_lut_step", self.int16_step)
                    if current_step is None or current_step <= 0:
                        current_step = 256
                    if current_step > 0:
                        lut = self.calculate_lut(op, info, 2**15 - 1, -(2**15), int(current_step))
                if lut != None:
                    lut_name = self.get_lut_name(op, info)
                    op.attributes["lut"] = lut_name
                    info.add_lut(lut_name, lut, info.get_var_exponents(op.outputs[0].name))
            return op
        AddLUTPattern.export = patched_export
    except ImportError:
        pass


    # -- Initialize esp_ppq_lut --
    import esp_ppq_lut as esp_lut
    esp_lut.initialize(step=LUT_STEP, verbose=True)
    from esp_ppq_lut import set_simulation_mode, SimulationMode

    # -- Seeds & Patches --
    seed_everything(QATConfig.SEED)
    register_mod_op()
    patch_v8_detection_loss()
    apply_esp_ppq_patches()

    # ==========================================
    # 3. LOAD MODEL & DATA
    # ==========================================
    def extract_model_meta():
        tmp_model = ESP_YOLO(QATConfig.PT_FILE)
        detect_head = tmp_model.model.model[-1]
        ch = [m[0].conv.in_channels for m in detect_head.cv2]
        meta = {
            'nc': detect_head.nc,
            'reg_max': detect_head.reg_max,
            'stride': detect_head.stride.tolist() if isinstance(detect_head.stride, torch.Tensor) else detect_head.stride,
            'ch': ch
        }
        return meta

    model_meta = extract_model_meta()
    data_cfg = check_det_dataset(QATConfig.DATA_YAML_FILE)

    # We need a dummy graph to initialize the trainer, but we will immediately override it
    # Loading the ONNX graph is the fastest way to get a valid graph object
    if not os.path.exists(QATConfig.ONNX_PATH):
        # Fallback to exporting it if missing
        model = ESP_YOLO(QATConfig.PT_FILE)
        apply_export_patches(model)
        model.export(format="onnx", opset=QATConfig.EXPORT_OPSET, simplify=True, imgsz=QATConfig.IMG_SZ)
        
    graph = load_onnx_graph(onnx_import_file=QATConfig.ONNX_PATH)

    # ==========================================
    # 4. INITIALIZE TRAINER & LOAD BEST GRAPH
    # ==========================================
    from trainer import QATTrainer
    
    trainer = QATTrainer(graph=graph, model_meta=model_meta, device=QATConfig.DEVICE)
    
    BEST_MODEL_PATH = os.path.join(QATConfig.ESPDL_OUTPUT_DIR, "Best_yolo26n.native")
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"ERROR: Could not find best model at {BEST_MODEL_PATH}")
        sys.exit(1)
        
    print(f"\nReloading Best Graph from: {BEST_MODEL_PATH}")
    trainer.load_graph(BEST_MODEL_PATH)
    
    # ==========================================
    # 5. VALIDATION ONLY
    # ==========================================
    print("\n" + "=" * 60)
    print("  5. STANDALONE VALIDATION")
    print("=" * 60)
    
    print("Running Validation on Best Saved Graph...")
    best_mAP = trainer.eval()

    print(f"\n--- Final Results ---")
    print(f"Best Model mAP50-95: {best_mAP:.3f}")
    
    print("\nValidation complete. Exiting.")
    sys.exit(0)
