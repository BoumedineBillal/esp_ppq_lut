"""
YOLO26n PTQ Dual-Mode Validation Test
======================================
Single-file pipeline: ONNX export → PTQ → LUT fusion → export → firmware test.
Proves esp_ppq_lut SIMULATION matches ESP32-P4 hardware bit-exactly for YOLO26n.
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
        EPOCHS = 8
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
    # The stock esp-ppq Exporter often defaults to step=256 even when kwargs are passed.
    # This patch forces AddLUTPattern to respect the operation-specific 'int16_lut_step'
    # attribute which we set correctly during the Fusion Pass.
    try:
        from esp_ppq.parser.espdl.export_patterns import (
            AddLUTPattern, Operation, BaseGraph, QuantableOperation,
            EspQuantType, ExporterPatternInfo
        )
        
        # Save original method just in case
        _original_export = AddLUTPattern.export

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
                    # PATCH: Prioritize op-specific step logic
                    # If op has 'int16_lut_step', use it. Otherwise fall back to exporter default.
                    current_step = op.attributes.get("int16_lut_step", self.int16_step)
                    
                    # Sanity check: Ensure step is valid
                    if current_step is None or current_step <= 0:
                        current_step = 256
                        
                    if current_step > 0:
                        lut = self.calculate_lut(op, info, 2**15 - 1, -(2**15), int(current_step))

                if lut != None:
                    lut_name = self.get_lut_name(op, info)
                    op.attributes["lut"] = lut_name
                    info.add_lut(lut_name, lut, info.get_var_exponents(op.outputs[0].name))

            return op

        print("[MonkeyPatch] Applying fix to AddLUTPattern.export for correct LUT step propagation...")
        AddLUTPattern.export = patched_export

    except ImportError as e:
        print(f"[MonkeyPatch] WARNING: Failed to patch AddLUTPattern: {e}")


    # -- Initialize esp_ppq_lut --
    import esp_ppq_lut as esp_lut
    esp_lut.initialize(step=LUT_STEP, verbose=True)
    from esp_ppq_lut import set_simulation_mode, SimulationMode

    # -- Seeds & Patches --
    seed_everything(QATConfig.SEED)
    register_mod_op()
    patch_v8_detection_loss()
    apply_esp_ppq_patches()

    print("Initialization complete.")


    # ==========================================
    # 3. ONNX EXPORT
    # ==========================================
    def extract_model_meta():
        tmp_model = ESP_YOLO(QATConfig.PT_FILE)
        detect_head = tmp_model.model.model[-1]
        ch = [m[0].conv.in_channels for m in detect_head.cv2]
        meta = {
            'nc': detect_head.nc,
            'reg_max': detect_head.reg_max,
            'stride': detect_head.stride,
            'ch': ch
        }
        if isinstance(meta['stride'], torch.Tensor):
            meta['stride'] = meta['stride'].tolist()
        print(f"Model Metadata: NC={meta['nc']}, RegMax={meta['reg_max']}, Stride={meta['stride']}")
        return meta


    def prepare_onnx():
        model = ESP_YOLO(QATConfig.PT_FILE)
        apply_export_patches(model)
        model.export(
            format="onnx", opset=QATConfig.EXPORT_OPSET,
            simplify=True, imgsz=QATConfig.IMG_SZ,
            dynamic=QATConfig.EXPORT_DYNAMIC
        )
        print(f"ONNX exported to: {QATConfig.ONNX_PATH}")


    print("\n" + "=" * 60)
    print("  3. ONNX EXPORT")
    print("=" * 60)
    prepare_onnx()
    model_meta = extract_model_meta()


    # ==========================================
    # 4. QUANTIZATION
    # ==========================================
    print("\n" + "=" * 60)
    print("  4. QUANTIZATION")
    print("=" * 60)

    graph = load_onnx_graph(onnx_import_file=QATConfig.ONNX_PATH)

    # -- Identify Aux vs Main branches --
    output_names = list(graph.outputs.keys())
    aux_ops = set()
    main_ops = set()

    if len(output_names) >= 6:
        aux_outputs = output_names[0:3]
        main_outputs = output_names[3:6]
        print(f"Identifying branches...")
        aux_ops = get_exclusive_ancestors(graph, aux_outputs, main_outputs)
        main_ops = get_exclusive_ancestors(graph, main_outputs, aux_outputs)
        print(f"  Aux-exclusive: {len(aux_ops)} ops")
        print(f"  Main-exclusive: {len(main_ops)} ops")
    else:
        print("WARNING: Graph output count < 6. Cannot separate aux/main branches.")

    # -- Set up dispatching --
    quantizer = PFL.Quantizer(platform=QATConfig.TARGET_PLATFORM, graph=graph)
    dispatching_table = PFL.Dispatcher(graph=graph, method="conservative").dispatch(
        quantizer.quant_operation_types
    )

    for opname, platform in dispatching_table.items():
        if platform == TargetPlatform.UNSPECIFIED:
            dispatching_table[opname] = TargetPlatform(quantizer.target_platform)

    # Disable quantization for Aux branch
    for op in aux_ops:
        if op.name in dispatching_table:
            dispatching_table[op.name] = TargetPlatform.FP32

    # -- INT16 Layer Configuration (Box + Class Head Conv+Swish) --
    INT16_PLATFORM = get_target_platform("esp32" + PLATFORM, 16)

    int16_layers = {
        # === ONE-TO-ONE BOX HEAD (one2one_cv2) ===
        # P3
        "/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv",
        "/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv/Swish",
        "/model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv",
        "/model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv/Swish",
        # "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv",            # ← Final 1×1 projection (stays INT8)
        # P4
        "/model.23/one2one_cv2.1/one2one_cv2.1.0/conv/Conv",
        "/model.23/one2one_cv2.1/one2one_cv2.1.0/conv/Conv/Swish",
        "/model.23/one2one_cv2.1/one2one_cv2.1.1/conv/Conv",
        "/model.23/one2one_cv2.1/one2one_cv2.1.1/conv/Conv/Swish",
        # "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv",            # ← Final 1×1 projection (stays INT8)
        # P5
        "/model.23/one2one_cv2.2/one2one_cv2.2.0/conv/Conv",
        "/model.23/one2one_cv2.2/one2one_cv2.2.0/conv/Conv/Swish",
        "/model.23/one2one_cv2.2/one2one_cv2.2.1/conv/Conv",
        "/model.23/one2one_cv2.2/one2one_cv2.2.1/conv/Conv/Swish",
        # "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv",            # ← Final 1×1 projection (stays INT8)

        # === ONE-TO-ONE CLASS HEAD (one2one_cv3) ===
        # P3
        "/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv",
        "/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv/Swish",
        "/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/conv/Conv",
        "/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/conv/Conv/Swish",
        "/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv",
        "/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv/Swish",
        "/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv",
        "/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv/Swish",
        # "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv",            # ← Final 1×1 projection (stays INT8)
        # P4
        "/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv",
        "/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv/Swish",
        "/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/conv/Conv",
        "/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/conv/Conv/Swish",
        "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv",
        "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv/Swish",
        "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv",
        "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv/Swish",
        # "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv",            # ← Final 1×1 projection (stays INT8)
        # P5
        "/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/conv/Conv",
        "/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/conv/Conv/Swish",
        "/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/conv/Conv",
        "/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/conv/Conv/Swish",
        "/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv",
        "/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv/Swish",
        "/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv",
        "/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv/Swish",
        # "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv",            # ← Final 1×1 projection (stays INT8)
    }

    # Apply INT16 to selected layers, default INT8 for rest
    for op in main_ops:
        if op.name in dispatching_table:
            if op.name in int16_layers:
                dispatching_table[op.name] = INT16_PLATFORM

    # Force Concat nodes to FP32 (known issue)
    fp32_layers = {
        "/model.23/Concat_5",
        "/model.23/Concat_3",
        "/model.23/Concat_4"
    }
    for op in main_ops:
        if op.name in fp32_layers:
            dispatching_table[op.name] = TargetPlatform.FP32
            print(f"  FP32: {op.name}")

    # Apply quantization
    print("Applying quantization...")
    for op in graph.operations.values():
        quantizer.quantize_operation(op_name=op.name, platform=dispatching_table[op.name])
    print("Quantization configured.")


    # ==========================================
    # 5. CALIBRATION (Phase 1 — without LUT fusion)
    # ==========================================
    print("\n" + "=" * 60)
    print("  5. CALIBRATION (Phase 1 — standard passes, no LUT)")
    print("=" * 60)

    data_cfg = check_det_dataset(QATConfig.DATA_YAML_FILE)
    cali_loader = get_calibration_loader(data_cfg)

    executor = TorchExecutor(graph=graph)
    dummy_input = torch.zeros([1, 3, IMG_SZ, IMG_SZ]).to(QATConfig.DEVICE)
    executor.tracing_operation_meta(inputs=dummy_input)

    print("Running Calibration Pipeline (no LUT)...")
    pipeline_pre_lut = PFL.Pipeline([
        QuantizeSimplifyPass(),
        QuantizeFusionPass(activation_type=quantizer.activation_fusion_types),
        ParameterQuantizePass(),
        RuntimeCalibrationPass(method=QATConfig.QUANT_CALIB_METHOD),
        PassiveParameterQuantizePass(clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE),
        QuantAlignmentPass(elementwise_alignment=QATConfig.QUANT_ALIGNMENT),
        # NOTE: No EspdlLUTFusionPass here — applied AFTER IDEAL export
    ])

    pipeline_pre_lut.optimize(
        calib_steps=QATConfig.CALIB_STEPS,
        collate_fn=(lambda x: x.type(torch.float).to(QATConfig.DEVICE)),
        graph=graph,
        dataloader=cali_loader,
        executor=executor,
    )
    print("Calibration Phase 1 Done (graph has standard Swish, no LUT tables).")

    # ==========================================
    # 5b. BASELINE VALIDATION (PTQ)
    # ==========================================
    from trainer import QATTrainer

    print("\n" + "=" * 60)
    print("  5b. BASELINE VALIDATION (PTQ)")
    print("=" * 60)
    print("Initializing Trainer for Baseline Check...")
    trainer = QATTrainer(graph=graph, model_meta=model_meta, device=QATConfig.DEVICE)

    print("Running Baseline Validation on Quantized Graph...")
    ptq_mAP = trainer.eval()

    print(f"\n--- Baseline Results ---")
    print(f"PTQ mAP50-95: {ptq_mAP:.3f}")


    # ==========================================
    # 5c. QAT TRAINING LOOP
    # ==========================================
    print("\n" + "=" * 60)
    print("  5c. QAT TRAINING LOOP")
    print("=" * 60)
    
    from dataset import get_train_loader
    train_loader = get_train_loader(data_cfg)

    print("Starting QAT Training...")
    best_mAP = ptq_mAP
    
    # Optional: Save baseline as initial best model
    trainer.save_graph(os.path.join(QATConfig.ESPDL_OUTPUT_DIR, "Best_yolo26n.native"))

    for epoch in range(QATConfig.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{QATConfig.EPOCHS} ---")
        
        # Train Epoch
        trainer.epoch(train_loader)
        
        # Validate
        current_mAP = trainer.eval()
        print(f"Epoch: {epoch+1}, mAP50-95: {current_mAP:.3f}")
        
        if current_mAP > best_mAP:
            best_mAP = current_mAP
            print(f"New best mAP! Saving to {QATConfig.ESPDL_OUTPUT_DIR}...")
            trainer.save_graph(os.path.join(QATConfig.ESPDL_OUTPUT_DIR, "Best_yolo26n.native"))

    # Load Best Graph back into memory at the end
    print("Reloading Best Graph for Inference Preparation...")
    trainer.load_graph(os.path.join(QATConfig.ESPDL_OUTPUT_DIR, "Best_yolo26n.native"))
    graph = trainer.graph 

    # ==========================================
    # 6. GRAPH SURGERY (Prepare for Inference)
    # ==========================================
    print("\n" + "=" * 60)
    print("  6. GRAPH SURGERY")
    print("=" * 60)


    def prune_graph_safely(graph: BaseGraph) -> BaseGraph:
        """Removes disconnected operations and unused variables."""
        round_count = 0
        while True:
            ops_removed = 0
            vars_removed = 0

            # Find dead ops (no consumers, not an output)
            dead_ops = []
            for op in list(graph.operations.values()):
                is_output = any(var.name in graph.outputs for var in op.outputs)
                has_consumers = any(len(var.dest_ops) > 0 for var in op.outputs)
                if not is_output and not has_consumers:
                    dead_ops.append(op)

            for op in dead_ops:
                for var in list(op.inputs):
                    op.inputs.remove(var)
                    if op in var.dest_ops:
                        var.dest_ops.remove(op)
                graph.remove_operation(op, keep_coherence=False)
                ops_removed += 1

            # Find dead variables
            dead_vars = []
            for var in list(graph.variables.values()):
                if var.name in graph.inputs or var.name in graph.outputs:
                    continue
                if len(var.dest_ops) == 0:
                    dead_vars.append(var)

            for var in dead_vars:
                if var.name in graph.variables:
                    graph.variables.pop(var.name)
                    vars_removed += 1

            round_count += 1
            if ops_removed == 0 and vars_removed == 0:
                break

        print(f"  Pruning: {round_count} rounds")
        return graph


    # 1. Remove Aux Heads
    output_names = list(graph.outputs.keys())
    if len(output_names) >= 6:
        aux_heads = output_names[0:3]
        print(f"Removing Aux Heads: {aux_heads}")
        for name in aux_heads:
            if name in graph.outputs:
                graph.outputs.pop(name)
        prune_graph_safely(graph)

    # 2. Split Concat outputs → 6 separate tensors
    targets = ["one2one_p3", "one2one_p4", "one2one_p5"]
    collected_outputs = {}

    for target_name in targets:
        if target_name in graph.outputs:
            original_output_var = graph.variables[target_name]
            producer = original_output_var.source_op

            if producer and producer.type == "Concat":
                print(f"Splitting {target_name} at Concat ({producer.name})...")

                box_var = None
                cls_var = None

                for input_var in producer.inputs:
                    dims = input_var.shape
                    if dims is not None:
                        if 4 in dims:
                            box_var = input_var
                        elif model_meta['nc'] in dims:
                            cls_var = input_var

                if box_var and cls_var:
                    pair_config = [
                        (box_var, f"{target_name}_box"),
                        (cls_var, f"{target_name}_cls"),
                    ]

                    for var, new_name in pair_config:
                        old_name = var.name
                        if old_name in graph.variables:
                            graph.variables.pop(old_name)
                        var._name = new_name
                        graph.variables[new_name] = var
                        collected_outputs[new_name] = var

                    graph.outputs.pop(target_name)

                    # Remove Concat op
                    graph.remove_operation(producer, keep_coherence=False)
                    for var in producer.inputs:
                        if producer in var.dest_ops:
                            var.dest_ops.remove(producer)

                    print(f"  -> {pair_config[0][1]} + {pair_config[1][1]}")
                else:
                    print(f"ERROR: Shape mismatch for {target_name}")
            else:
                print(f"WARNING: Source for {target_name} is not Concat.")

    # 3. Register outputs in strict order
    final_output_list = [
        "one2one_p3_box", "one2one_p3_cls",
        "one2one_p4_box", "one2one_p4_cls",
        "one2one_p5_box", "one2one_p5_cls"
    ]

    graph.outputs.clear()
    for name in final_output_list:
        if name in collected_outputs:
            graph.outputs[name] = collected_outputs[name]

    # 4. Final prune
    prune_graph_safely(graph)
    print(f"Final Graph Outputs: {list(graph.outputs.keys())}")


    # ==========================================
    # 7a. EXPORT IDEAL_MATH MODEL (no LUT)
    # ==========================================
    print("\n" + "=" * 60)
    print("  7a. EXPORT IDEAL_MATH MODEL (no LUT)")
    print("=" * 60)

    espdl_ideal_path = os.path.join(OUTPUT_DIR, f"yolo26n_{IMG_SZ}_ideal_{PLATFORM}.espdl")
    print(f"Exporting IDEAL model to: {espdl_ideal_path}")

    exporter = PFL.Exporter(platform=QATConfig.TARGET_PLATFORM)
    exporter.export(espdl_ideal_path, graph=graph)
    print(f"IDEAL_MATH .espdl export complete: {espdl_ideal_path}")


    # ==========================================
    # 7b. APPLY LUT FUSION (Phase 2) + EXPORT LUT MODEL
    # ==========================================
    print("\n" + "=" * 60)
    print("  7b. LUT FUSION + EXPORT LUT MODEL")
    print("=" * 60)

    lut_pass = esp_lut.EspdlLUTFusionPass(
        target_ops=['Swish'],
        verify=False,
        output_dir=OUTPUT_DIR,
        lut_step=LUT_STEP
    )

    # Apply LUT fusion to the already-calibrated graph
    # (replaces Swish ops with LUT tables in-place)
    lut_pass.optimize(
        graph=graph,
        dataloader=cali_loader,
        executor=TorchExecutor(graph=graph),
        calib_steps=0,
        collate_fn=(lambda x: x.type(torch.float).to(QATConfig.DEVICE)),
    )
    print("LUT fusion applied.")

    espdl_lut_path = os.path.join(OUTPUT_DIR, f"yolo26n_{IMG_SZ}_s8_{PLATFORM}.espdl")
    print(f"Exporting LUT model to: {espdl_lut_path}")

    exporter_lut = PFL.Exporter(platform=QATConfig.TARGET_PLATFORM)
    exporter_lut.export(espdl_lut_path, graph=graph, int16_lut_step=LUT_STEP)
    print(f"LUT .espdl export complete: {espdl_lut_path}")


    # ==========================================
    # 8. GENERATE TEST VECTORS
    # ==========================================
    print("\n" + "=" * 60)
    print("  8. GENERATE TEST VECTORS")
    print("=" * 60)

    # Use person.jpg as absolute deterministic test input
    test_image_path = os.path.join(SCRIPT_DIR, "images", "person.jpg")
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Test image not found at {test_image_path}")

    def espdl_preprocess(img_bgr, dst_shape, pad_val=114):
        """
        Geometrically exact clone of ESP-DL's C++ `resize_nn` & padding logic.
        """
        src_h, src_w = img_bgr.shape[:2]
        dst_h, dst_w = dst_shape
        
        scale_x = dst_w / float(src_w)
        scale_y = dst_h / float(src_h)
        scale = min(scale_x, scale_y)
        
        border_top, border_bottom, border_left, border_right = 0, 0, 0, 0
        if scale_x < scale_y:
            pad_h = dst_h - int(scale * src_h)
            border_top = pad_h // 2
            border_bottom = pad_h - border_top
        else:
            pad_w = dst_w - int(scale * src_w)
            border_left = pad_w // 2
            border_right = pad_w - border_left
            
        act_dst_w = dst_w - border_left - border_right
        act_dst_h = dst_h - border_top - border_bottom
        
        inv_scale_x = float(src_w) / act_dst_w
        inv_scale_y = float(src_h) / act_dst_h
        
        out_img = np.full((dst_h, dst_w, 3), pad_val, dtype=np.uint8)
        
        for y_dst in range(act_dst_h):
            y_src = min(int(y_dst * inv_scale_y), src_h - 1)
            for x_dst in range(act_dst_w):
                x_src = min(int(x_dst * inv_scale_x), src_w - 1)
                out_img[y_dst + border_top, x_dst + border_left] = img_bgr[y_src, x_src]
                
        return out_img

    # Simulate Python PyTorch Preprocessing (Exact ESP-DL Match)
    im0 = cv2.imread(test_image_path)
    im = espdl_preprocess(im0, dst_shape=(QATConfig.IMG_SZ, QATConfig.IMG_SZ))
    im_draw = im.copy()

    im = im[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, HWC to CHW
    im = np.ascontiguousarray(im)

    test_input = torch.from_numpy(im).to(QATConfig.DEVICE).float() / 255.0
    test_input = test_input.unsqueeze(0)  # (1, 3, IMG_SZ, IMG_SZ)

    print(f"Test Input shape: {test_input.shape}, range: [{test_input.min():.4f}, {test_input.max():.4f}]")

    # A. SIMULATION mode output (should match firmware bit-exactly)
    set_simulation_mode(SimulationMode.SIMULATION)
    executor_sim = TorchExecutor(graph=graph)
    outputs_sim = executor_sim.forward(test_input)
    print(f"  SIMULATION: {len(outputs_sim)} outputs")

    # B. IDEAL_MATH mode output (expected mismatches)
    set_simulation_mode(SimulationMode.IDEAL_MATH)
    executor_ideal = TorchExecutor(graph=graph)
    outputs_ideal = executor_ideal.forward(test_input)
    print(f"  IDEAL_MATH: {len(outputs_ideal)} outputs")

    # Revert
    set_simulation_mode(SimulationMode.SIMULATION)

    # Map outputs to names
    output_keys = list(graph.outputs.keys())
    assert len(output_keys) == len(outputs_sim) == len(outputs_ideal), \
        f"Output count mismatch: keys={len(output_keys)}, sim={len(outputs_sim)}, ideal={len(outputs_ideal)}"

    test_vectors = {}
    for i, name in enumerate(output_keys):
        sim_tensor = outputs_sim[i].detach().cpu()
        ideal_tensor = outputs_ideal[i].detach().cpu()
        test_vectors[name] = {
            'sim': sim_tensor,
            'ideal': ideal_tensor,
            'shape': list(sim_tensor.shape),
            'size': sim_tensor.numel(),
        }
        print(f"  {name}: shape={list(sim_tensor.shape)}, size={sim_tensor.numel()}, "
            f"sim_range=[{sim_tensor.min():.4f}, {sim_tensor.max():.4f}]")

    # ==========================================
    # 8b. POST-PROCESS & PLOT LUT SIMULATION
    # ==========================================
    def decode_and_draw(outputs_dict, image_bgr, conf_thresh=0.25):
        import matplotlib.pyplot as plt
        strides = [8, 16, 32]
        targets = ["p3", "p4", "p5"]
        all_boxes = []
        all_scores = []
        
        for i in range(3):
            stride = strides[i]
            box = outputs_dict[f'one2one_{targets[i]}_box']['sim'].float() # (1, 4, H, W)
            cls = outputs_dict[f'one2one_{targets[i]}_cls']['sim'].float() # (1, NC, H, W)
            
            _, _, H, W = box.shape
            # Create grid
            grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            grid = torch.stack((grid_x, grid_y), dim=0).float() + 0.5  # (2, H, W)
            grid = grid.to(box.device).unsqueeze(0) # (1, 2, H, W)
            
            # d_l, d_t, d_r, d_b
            dl = box[:, 0:1, :, :]
            dt = box[:, 1:2, :, :]
            dr = box[:, 2:3, :, :]
            db = box[:, 3:4, :, :]
            
            x1 = (grid[:, 0:1, :, :] - dl) * stride
            y1 = (grid[:, 1:2, :, :] - dt) * stride
            x2 = (grid[:, 0:1, :, :] + dr) * stride
            y2 = (grid[:, 1:2, :, :] + db) * stride
            
            decoded_boxes = torch.cat([x1, y1, x2, y2], dim=1) # (1, 4, H, W)
            all_boxes.append(decoded_boxes.flatten(2)) # (1, 4, H*W)
            all_scores.append(cls.flatten(2))          # (1, NC, H*W)
            
        all_boxes = torch.cat(all_boxes, dim=-1) # (1, 4, 8400)
        all_scores = torch.cat(all_scores, dim=-1) # (1, NC, 8400)
        
        # Sigmoid on raw logits
        all_scores = all_scores.sigmoid()
        max_scores, class_ids = torch.max(all_scores, dim=1) # (1, 8400)
        
        # Filter by confidence
        mask = max_scores > conf_thresh
        final_boxes = all_boxes[0, :, mask[0]].T # (N, 4)
        final_scores = max_scores[0, mask[0]] # (N,)
        final_classes = class_ids[0, mask[0]] # (N,)
        
        img_plot = image_bgr.copy()
        
        print(f"\\nFound {len(final_boxes)} detections from LUT Simulation:")
        for i in range(len(final_boxes)):
            x1, y1, x2, y2 = map(int, final_boxes[i])
            score = final_scores[i].item()
            cid = final_classes[i].item()
            print(f"  - Class {cid}, Score {score:.2f}, Box [{x1}, {y1}, {x2}, {y2}]")
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_plot.shape[1], x2), min(img_plot.shape[0], y2)
            
            cv2.rectangle(img_plot, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_plot, f"Class {cid} {score:.2f}", (x1, max(y1-10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Plot using Matplotlib
        plt.figure("LUT Simulation Detections", figsize=(8,8))
        plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("PPQ LUT INT16 Simulation Detections")
        plt.show()

    decode_and_draw(test_vectors, im_draw, conf_thresh=0.25)



    # ==========================================
    # 9. GENERATE FIRMWARE PROJECT
    # ==========================================
    print("\n" + "=" * 60)
    print("  9. GENERATE FIRMWARE PROJECT")
    print("=" * 60)

    FIRMWARE_DIR = os.path.join(SCRIPT_DIR, "firmware")
    FIRMWARE_MAIN_DIR = os.path.join(FIRMWARE_DIR, "main")
    FIRMWARE_MODELS_DIR = os.path.join(FIRMWARE_MAIN_DIR, "models")
    FIRMWARE_TEST_DATA_DIR = os.path.join(FIRMWARE_MAIN_DIR, "test_data")

    os.makedirs(FIRMWARE_MODELS_DIR, exist_ok=True)
    os.makedirs(FIRMWARE_TEST_DATA_DIR, exist_ok=True)


    # --- A. Write test data header ---
    def write_float_array(f, var_name, tensor):
        """Writes a float array to an open header file."""
        flat = tensor.flatten().numpy()
        f.write(f"const int {var_name}_len = {len(flat)};\n\n")
        f.write(f"const float {var_name}[] = {{\n")
        for i, val in enumerate(flat):
            f.write(f"    {repr(float(val))}f")
            if i < len(flat) - 1:
                f.write(",")
            if (i + 1) % 8 == 0:
                f.write("\n")
        f.write(f"\n}};\n\n")


    header_path = os.path.join(FIRMWARE_TEST_DATA_DIR, f"yolo_test_{PLATFORM}_data.h")

    with open(header_path, 'w') as f:
        f.write(f"// Auto-generated test data for YOLO26n dual-mode validation\n")
        f.write(f"// IMG_SZ: {IMG_SZ}, LUT_STEP: {LUT_STEP}, PLATFORM: {PLATFORM}\n")
        f.write(f"// Outputs: {output_keys}\n\n")
        f.write(f"#pragma once\n\n")

        # Input data
        # Transpose NCHW -> NHWC to match ESP-DL input layout
        in_nhwc = test_input.cpu().permute(0, 2, 3, 1).contiguous()
        in_flat = in_nhwc.flatten().numpy()
        f.write(f"const int test_input_len = {len(in_flat)};\n\n")
        f.write(f"const float test_input[] = {{\n")
        for i, val in enumerate(in_flat):
            f.write(f"    {val:.8f}f")
            if i < len(in_flat) - 1:
                f.write(",")
            if (i + 1) % 8 == 0:
                f.write("\n")
        f.write(f"\n}};\n\n")

        # Output count
        f.write(f"const int num_outputs = {len(output_keys)};\n\n")

        # Output names (for logging)
        f.write(f"const char* output_names[] = {{\n")
        for i, name in enumerate(output_keys):
            comma = "," if i < len(output_keys) - 1 else ""
            f.write(f'    "{name}"{comma}\n')
        f.write(f"}};\n\n")

        # Output sizes
        f.write(f"const int output_sizes[] = {{\n    ")
        f.write(", ".join([str(test_vectors[name]['size']) for name in output_keys]))
        f.write(f"\n}};\n\n")

        # SIMULATION + IDEAL_MATH arrays for each output
        for name in output_keys:
            safe_name = name.replace("/", "_").replace(".", "_")
            # Transpose NCHW -> NHWC to match ESP-DL output layout
            sim_nhwc = test_vectors[name]['sim'].permute(0, 2, 3, 1).contiguous()
            ideal_nhwc = test_vectors[name]['ideal'].permute(0, 2, 3, 1).contiguous()
            write_float_array(f, f"output_sim_{safe_name}", sim_nhwc)
            write_float_array(f, f"output_ideal_{safe_name}", ideal_nhwc)

        # Pointer arrays for indexed access
        f.write(f"const float* output_sim_ptrs[] = {{\n")
        for i, name in enumerate(output_keys):
            safe_name = name.replace("/", "_").replace(".", "_")
            comma = "," if i < len(output_keys) - 1 else ""
            f.write(f"    output_sim_{safe_name}{comma}\n")
        f.write(f"}};\n\n")

        f.write(f"const float* output_ideal_ptrs[] = {{\n")
        for i, name in enumerate(output_keys):
            safe_name = name.replace("/", "_").replace(".", "_")
            comma = "," if i < len(output_keys) - 1 else ""
            f.write(f"    output_ideal_{safe_name}{comma}\n")
        f.write(f"}};\n\n")

    header_size_kb = os.path.getsize(header_path) / 1024
    print(f"  Test data header -> {header_path} ({header_size_kb:.0f} KB)")

    # --- A2. Write Raw RGB data to bypass C++ JPEG Decoder ---
    im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    in_flat_rgb = im0_rgb.flatten().tolist()
    header_raw_path = os.path.join(FIRMWARE_TEST_DATA_DIR, f"raw_rgb_person.h")
    with open(header_raw_path, 'w') as f:
        f.write(f"const int raw_rgb_person_len = {len(in_flat_rgb)};\n\n")
        f.write(f"const uint8_t raw_rgb_person[] = {{\n")
        for i in range(0, len(in_flat_rgb), 16):
            f.write("    " + ",".join([str(val) for val in in_flat_rgb[i:i+16]]))
            if i + 16 < len(in_flat_rgb):
                f.write(",\n")
        f.write(f"\n}};\n\n")
    print(f"  Raw RGB header -> {header_raw_path} ({os.path.getsize(header_raw_path) / 1024:.0f} KB)")


    # --- B. Copy BOTH .espdl models ---
    dst_model_lut = os.path.join(FIRMWARE_MODELS_DIR, f"yolo26n_{PLATFORM}.espdl")
    dst_model_ideal = os.path.join(FIRMWARE_MODELS_DIR, f"yolo26n_ideal_{PLATFORM}.espdl")
    shutil.copy2(espdl_lut_path, dst_model_lut)
    shutil.copy2(espdl_ideal_path, dst_model_ideal)
    print(f"  LUT model   -> {dst_model_lut}")
    print(f"  IDEAL model -> {dst_model_ideal}")


    # --- C. Generate main.cpp (3-test firmware) ---
    main_cpp_path = os.path.join(FIRMWARE_MAIN_DIR, "main.cpp")

    model_lut_asm = f"_binary_yolo26n_{PLATFORM}_espdl_start"
    model_ideal_asm = f"_binary_yolo26n_ideal_{PLATFORM}_espdl_start"

    with open(main_cpp_path, 'w', encoding='utf-8') as f:
        f.write(f'''/*
    * YOLO26n PTQ Triple-Mode Validation Firmware
    * IMG_SZ: {IMG_SZ}, LUT_STEP: {LUT_STEP}
    *
    * Model A: LUT model   (Swish replaced by LUT tables)
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
    #include "nvs_flash.h"
    #include "esp_log.h"

    #include "dl_model_base.hpp"
    #include "dl_tensor_base.hpp"
    #include "dl_tool.hpp"
    #include "yolo26.hpp"

    // Test Data
    namespace test_data {{
    #include "test_data/yolo_test_{PLATFORM}_data.h"
    #include "test_data/raw_rgb_person.h"
    }}

    static const char *TAG = "YOLO26_VAL";

    // Model Binary
    extern const uint8_t model_lut_bin[]   asm("{model_lut_asm}");
    extern const uint8_t model_ideal_bin[] asm("{model_ideal_asm}");

    // Comparison result for a single output
    struct CompareResult {{
        int mismatches;   // count of elements where |diff| > 0
        int fail_tol;     // count of elements where |diff| > 1 (exceed tolerance)
        int min_err;      // minimum signed error (most negative)
        int max_err;      // maximum signed error (most positive)
    }};

    /**
    * Compare HW output against a reference output array (INT8).
    */
    CompareResult compare_output_int8(int8_t *hw_ptr, const float *ref, int size,
                                    float out_scale, const char *label)
    {{
        CompareResult r = {{0, 0, 0, 0}};
        for (int i = 0; i < size; i++) {{
            int rounded_out = dl::tool::round(ref[i] / out_scale);
            if (rounded_out > 127) rounded_out = 127;
            if (rounded_out < -128) rounded_out = -128;
            int8_t ref_int = (int8_t)rounded_out;
            int8_t hw_int = hw_ptr[i];
            int diff = (int)hw_int - (int)ref_int;
            if (diff != 0) {{
                if (r.mismatches < 3)
                    ESP_LOGW(TAG, "    [%d]: HW=%d vs %s=%d (diff=%d)",
                            i, hw_int, label, ref_int, diff);
                r.mismatches++;
                if (abs(diff) > 1) r.fail_tol++;
                if (diff < r.min_err) r.min_err = diff;
                if (diff > r.max_err) r.max_err = diff;
            }}
        }}
        return r;
    }}

    /**
    * Compare HW output against a reference output array (INT16).
    */
    CompareResult compare_output_int16(int16_t *hw_ptr, const float *ref, int size,
                                    float out_scale, const char *label)
    {{
        CompareResult r = {{0, 0, 0, 0}};
        for (int i = 0; i < size; i++) {{
            int rounded_out = dl::tool::round(ref[i] / out_scale);
            if (rounded_out > 32767) rounded_out = 32767;
            if (rounded_out < -32768) rounded_out = -32768;
            int16_t ref_int = (int16_t)rounded_out;
            int16_t hw_int = hw_ptr[i];
            int diff = (int)hw_int - (int)ref_int;
            if (diff != 0) {{
                if (r.mismatches < 3)
                    ESP_LOGW(TAG, "    [%d]: HW=%d vs %s=%d (diff=%d)",
                            i, hw_int, label, ref_int, diff);
                r.mismatches++;
                if (abs(diff) > 1) r.fail_tol++;
                if (diff < r.min_err) r.min_err = diff;
                if (diff > r.max_err) r.max_err = diff;
            }}
        }}
        return r;
    }}

    /**
    * Inject test input into model and run inference.
    */
    void inject_input_and_run(dl::Model *model, const char *model_name)
    {{
        auto &inputs = model->get_inputs();
        dl::TensorBase *input_tensor = inputs.begin()->second;
        float in_scale = powf(2.0f, input_tensor->exponent);
        int8_t *input_ptr = (int8_t *)input_tensor->data;
        int in_size = input_tensor->size;

        ESP_LOGI(TAG, "[%s] Input size: %d, scale: %.8f", model_name, in_size, in_scale);

        for (int i = 0; i < in_size && i < test_data::test_input_len; i++) {{
            float val = test_data::test_input[i] / in_scale;
            int rounded_val = dl::tool::round(val);
            if (rounded_val > 127) rounded_val = 127;
            if (rounded_val < -128) rounded_val = -128;
            input_ptr[i] = (int8_t)rounded_val;
        }}

        ESP_LOGI(TAG, "[%s] Running inference...", model_name);
        model->run();
    }}

    /**
    * Compare all outputs of a model against reference arrays.
    * For expect_match tests: PASS if all errors are within +/-1 tolerance.
    * Returns total mismatches.
    */
    int compare_all_outputs(dl::Model *model, const float **ref_ptrs, const char *ref_label,
                            bool expect_match)
    {{
        auto &outputs = model->get_outputs();
        int output_idx = 0;
        int total_mismatches = 0;
        int total_fail_tol = 0;
        int total_values = 0;
        int global_min_err = 0;
        int global_max_err = 0;

        for (auto &kv : outputs) {{
            const char *name = test_data::output_names[output_idx];
            dl::TensorBase *tensor = kv.second;
            float out_scale = powf(2.0f, tensor->exponent);
            int size = tensor->size;
            CompareResult r;

            if (tensor->exponent < -7) {{
                r = compare_output_int16((int16_t *)tensor->data,
                                        ref_ptrs[output_idx],
                                        size, out_scale, ref_label);
            }} else {{
                r = compare_output_int8((int8_t *)tensor->data,
                                        ref_ptrs[output_idx],
                                        size, out_scale, ref_label);
            }}

            if (expect_match) {{
                if (r.mismatches == 0) {{
                    ESP_LOGI(TAG, "  %s: PASS (%d values, err: 0)", name, size);
                }} else if (r.fail_tol == 0) {{
                    ESP_LOGI(TAG, "  %s: PASS +/-1 (%d/%d mismatches, err_range:[%d,%d])",
                            name, r.mismatches, size, r.min_err, r.max_err);
                }} else {{
                    ESP_LOGE(TAG, "  %s: FAIL %d/%d exceed +/-1 (%d/%d total, err_range:[%d,%d])",
                            name, r.fail_tol, size, r.mismatches, size, r.min_err, r.max_err);
                }}
            }} else {{
                if (r.mismatches > 0)
                    ESP_LOGI(TAG, "  %s: %d/%d mismatches (err_range:[%d,%d])",
                            name, r.mismatches, size, r.min_err, r.max_err);
                else
                    ESP_LOGW(TAG, "  %s: 0 mismatches (unexpected)", name);
            }}

            total_mismatches += r.mismatches;
            total_fail_tol += r.fail_tol;
            total_values += size;
            if (r.min_err < global_min_err) global_min_err = r.min_err;
            if (r.max_err > global_max_err) global_max_err = r.max_err;
            output_idx++;
        }}

        if (expect_match) {{
            if (total_mismatches == 0) {{
                ESP_LOGI(TAG, "  \\033[32mTOTAL PASS: 100%%%% Bit-Exact (%d values, %d outputs)\\033[0m",
                        total_values, output_idx);
            }} else if (total_fail_tol == 0) {{
                ESP_LOGI(TAG, "  \\033[32mTOTAL PASS within +/-1: %d/%d mismatches (err_range:[%d,%d], %d outputs)\\033[0m",
                        total_mismatches, total_values, global_min_err, global_max_err, output_idx);
            }} else {{
                ESP_LOGE(TAG, "  \\033[31mTOTAL FAIL: %d/%d exceed +/-1 (%d/%d total, err_range:[%d,%d])\\033[0m",
                        total_fail_tol, total_values, total_mismatches, total_values,
                        global_min_err, global_max_err);
            }}
        }} else {{
            ESP_LOGI(TAG, "  TOTAL: %d mismatches (err_range:[%d,%d])",
                    total_mismatches, global_min_err, global_max_err);
        }}

        return total_mismatches;
    }}

    extern "C" void app_main(void)
    {{
        ESP_LOGI(TAG, "==========================================================");
        ESP_LOGI(TAG, "   YOLO26n PTQ Triple-Mode Validation");
        ESP_LOGI(TAG, "   IMG_SZ: {IMG_SZ}, LUT_STEP: {LUT_STEP}");
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
        if (!model_lut) {{ ESP_LOGE(TAG, "Failed to load LUT model!"); return; }}

        // Native HW Preprocessing test
        YOLO26 processor(model_lut, 32, 0.25f);
        
        // BYPASS JPEG DECODER: Feed raw uncompressed RGB to ImagePreprocessor
        dl::image::img_t img;
        img.data = (uint8_t*)test_data::raw_rgb_person;
        img.width = 500;
        img.height = 375;
        img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;

        processor.preprocess(img);
        
        auto &inputs = model_lut->get_inputs();
        dl::TensorBase *input_tensor = inputs.begin()->second;
        float in_scale = powf(2.0f, input_tensor->exponent);
        int in_size = input_tensor->size;
        
        ESP_LOGI(TAG, "[LUT] Input size: %d, scale: %.8f", in_size, in_scale);
        ESP_LOGI(TAG, "----------------------------------------------------------");
        ESP_LOGI(TAG, "TEST 0: HW(LUT Preprocess) vs Python(test_input) — expect match (+/-1 tol)");
        
        CompareResult r_input = compare_output_int8((int8_t *)input_tensor->data,
                                                test_data::test_input,
                                                in_size, in_scale, "SIM_IN");
        if (r_input.fail_tol > 0) {{
            ESP_LOGE(TAG, "  input_tensor: FAIL %d/%d exceed +/-1 (%d/%d total, err_range:[%d,%d])", 
                    r_input.fail_tol, in_size, r_input.mismatches, in_size, r_input.min_err, r_input.max_err);
        }} else if (r_input.mismatches > 0) {{
            ESP_LOGI(TAG, "  input_tensor: PASS +/-1 (%d/%d mismatches, err_range:[%d,%d])", 
                    r_input.mismatches, in_size, r_input.min_err, r_input.max_err);
        }} else {{
            ESP_LOGI(TAG, "  input_tensor: PASS (%d values, err: 0)", in_size);
        }}
        
        // Do NOT free img.data because we assigned it directly to a const array in Flash ROM (.rodata)

        ESP_LOGI(TAG, "[LUT] Running inference...");
        model_lut->run();

        // TEST 1: HW(LUT model) vs SIMULATION vectors
        ESP_LOGI(TAG, "----------------------------------------------------------");
        ESP_LOGI(TAG, "TEST 1: HW(LUT model) vs SIMULATION — expect match (+/-1 tol)");
        compare_all_outputs(model_lut, test_data::output_sim_ptrs, "SIM", true);

        // TEST 2: HW(LUT model) vs IDEAL_MATH vectors
        ESP_LOGI(TAG, "----------------------------------------------------------");
        ESP_LOGI(TAG, "TEST 2: HW(LUT model) vs IDEAL_MATH — expect mismatches");
        compare_all_outputs(model_lut, test_data::output_ideal_ptrs, "IDEAL", false);

        // Print Final Post-Processing Bounding Boxes
        ESP_LOGI(TAG, "----------------------------------------------------------");
        ESP_LOGI(TAG, "Running YOLO26 Post-Processing...");
        auto results = processor.postprocess(model_lut->get_outputs());
        for(auto& res : results) {{
            ESP_LOGI("YOLO26", "[category: %s, score: %.2f, x1: %d, y1: %d, x2: %d, y2: %d]", 
                processor.class_names[res.class_id], res.score, (int)res.x1, (int)res.y1, (int)res.x2, (int)res.y2);
        }}

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
        if (!model_ideal) {{ ESP_LOGE(TAG, "Failed to load IDEAL model!"); return; }}

        inject_input_and_run(model_ideal, "IDEAL");

        // TEST 3: HW(IDEAL model) vs IDEAL_MATH vectors
        ESP_LOGI(TAG, "----------------------------------------------------------");
        ESP_LOGI(TAG, "TEST 3: HW(IDEAL model) vs IDEAL_MATH — expect match (+/-1 tol)");
        compare_all_outputs(model_ideal, test_data::output_ideal_ptrs, "IDEAL", true);

        delete model_ideal;

        ESP_LOGI(TAG, "==========================================================");
        ESP_LOGI(TAG, "Validation Finished.");
    }}
    ''')
    print(f"  main.cpp -> {main_cpp_path}")


    # --- D. Generate main/CMakeLists.txt ---
    main_cmake_path = os.path.join(FIRMWARE_MAIN_DIR, "CMakeLists.txt")
    with open(main_cmake_path, 'w') as f:
        f.write(f'''# Register the main component
    idf_component_register(SRCS "main.cpp"
                        PRIV_REQUIRES nvs_flash esp-dl yolo26)

    # Import esp-dl Utilities
    idf_build_get_property(component_targets __COMPONENT_TARGETS)
    if ("___idf_espressif__esp-dl" IN_LIST component_targets)
        idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
    else()
        idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
    endif()

    set(cmake_dir ${{espdl_dir}}/fbs_loader/cmake)
    include(${{cmake_dir}}/utilities.cmake)

    # Embed Both Models (LUT + IDEAL)
    target_add_aligned_binary_data(${{COMPONENT_LIB}} "models/yolo26n_{PLATFORM}.espdl" BINARY)
    target_add_aligned_binary_data(${{COMPONENT_LIB}} "models/yolo26n_ideal_{PLATFORM}.espdl" BINARY)
    ''')
    print(f"  CMakeLists.txt -> {main_cmake_path}")


    # --- E. Generate root CMakeLists.txt ---
    root_cmake_path = os.path.join(FIRMWARE_DIR, "CMakeLists.txt")
    with open(root_cmake_path, 'w') as f:
        f.write('''cmake_minimum_required(VERSION 3.16)
    include($ENV{IDF_PATH}/tools/cmake/project.cmake)
    project(yolo26n_validation)
    ''')
    print(f"  CMakeLists.txt -> {root_cmake_path}")


    # --- F. Generate partitions.csv ---
    partitions_path = os.path.join(FIRMWARE_DIR, "partitions.csv")
    with open(partitions_path, 'w') as f:
        f.write('''# Name,   Type, SubType, Offset,  Size, Flags
    factory,  app,  factory,  0x010000,  15000K,
    ''')
    print(f"  partitions.csv -> {partitions_path}")


    # --- G. Generate sdkconfig.defaults.esp32p4 ---
    sdkconfig_path = os.path.join(FIRMWARE_DIR, "sdkconfig.defaults.esp32p4")
    with open(sdkconfig_path, 'w') as f:
        f.write('''# YOLO26n Validation - ESP32-P4
    CONFIG_IDF_TARGET="esp32p4"
    CONFIG_ESPTOOLPY_FLASHMODE_QIO=y
    CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y
    CONFIG_PARTITION_TABLE_CUSTOM=y
    CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions.csv"
    CONFIG_SPIRAM=y
    CONFIG_SPIRAM_SPEED_200M=y
    CONFIG_CACHE_L2_CACHE_256KB=y
    CONFIG_CACHE_L2_CACHE_LINE_128B=y
    CONFIG_ESP_SYSTEM_ALLOW_RTC_FAST_MEM_AS_HEAP=n
    CONFIG_ESP_INT_WDT=n
    CONFIG_ESP_TASK_WDT_EN=n
    CONFIG_IDF_EXPERIMENTAL_FEATURES=y
    ''')
    print(f"  sdkconfig -> {sdkconfig_path}")


    # --- H. Generate empty sdkconfig.defaults ---
    sdkconfig_defaults_path = os.path.join(FIRMWARE_DIR, "sdkconfig.defaults")
    with open(sdkconfig_defaults_path, 'w') as f:
        f.write("")
    print(f"  sdkconfig.defaults -> {sdkconfig_defaults_path}")


    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 60)
    print("  OUTPUT FILES SUMMARY")
    print("=" * 60)
    for f_name in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f_name)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f_name:50s} {size_kb:8.1f} KB")
    print("=" * 60)

    print(f"\n✅ Firmware project generated at: {FIRMWARE_DIR}")
    print(f"   Next steps:")
    print(f"   1. cd {FIRMWARE_DIR}")
    print(f"   2. idf.py set-target esp32p4")
    print(f"   3. idf.py build")
    print(f"   4. idf.py flash monitor")
