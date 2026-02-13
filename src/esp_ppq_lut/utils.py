
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def update_verification_manifest(output_dir, index, metadata):
    """
    Updates the mapping.json file with new verification metadata.
    """
    import json
    manifest_path = os.path.join(output_dir, "mapping.json")
    
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except:
            pass
            
    manifest[f"lut_{index}"] = metadata
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

def run_numerical_verification(y_sim, y_ideal, title="Verification", output_dir="outputs", is_table=False, filename=None):
    """
    Performs numerical analysis between simulation and ideal results.
    If is_table=True, it expects integer tensors and performs bit-exact matching.
    """
    y_sim = y_sim.flatten().detach().cpu()
    y_ideal = y_ideal.flatten().detach().cpu()
    
    mse = F.mse_loss(y_sim.float(), y_ideal.float()).item()
    max_err = torch.max(torch.abs(y_sim.float() - y_ideal.float())).item()
    
    # For tables or bit-exact tests, tolerance is 0
    atol = 0 if is_table else 1e-7
    matched_elements = torch.isclose(y_sim.float(), y_ideal.float(), atol=atol)
    matches = matched_elements.float().mean().item() * 100
    
    # Optional: Check if error is within 1 LSB (Least Significant Bit)
    lsb_match = ""
    if not is_table:
        lsb_match = " (Expected for Quantized vs Smooth Math)"

    print(f"\n--- {title} ---")
    if is_table:
        print(f"Total Points: {len(y_sim)}")
        print(f"Bit-Exact:    {int(matched_elements.sum().item())}")
    print(f"MSE:          {mse:.10e}")
    print(f"Max Error:    {max_err:.10f}")
    print(f"Match %:      {matches:.2f}%{lsb_match}")
    
    # Plotting for all verification types
    plt.figure(figsize=(10, 5))
    indices = np.arange(len(y_sim))
    plt.plot(indices, y_sim.numpy(), label='Hardware Sim', color='red', alpha=0.8)
    plt.plot(indices, y_ideal.numpy(), label='Ideal Math', linestyle='--', color='blue', alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if filename:
        save_name = f"{filename}.png"
    else:
        # Fallback to sanitized title if no filename provided
        safe_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").lower()
        save_name = f"verify_{safe_title}.png"

    plt.savefig(os.path.join(output_dir, save_name))
    plt.close()

def generate_comparison_plot(x, y_sim, y_ideal, x_calib=None, y_calib=None, output_path="outputs/comparison.png"):
    """
    Generates an X-Y comparison plot matching validate_lut.py style.
    """
    x = x.flatten().detach().cpu().numpy()
    y_sim = y_sim.flatten().detach().cpu().numpy()
    y_ideal = y_ideal.flatten().detach().cpu().numpy()
    
    sort_idx = np.argsort(x)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x[sort_idx], y_ideal[sort_idx], label='Ideal Math (F32)', color='blue', alpha=0.3, linestyle='--')
    plt.plot(x[sort_idx], y_sim[sort_idx], label='Hardware LUT Sim', color='red', linewidth=1.5, alpha=0.8)
    
    if x_calib is not None and y_calib is not None:
        xc = x_calib.flatten().detach().cpu().numpy()
        yc = y_calib.flatten().detach().cpu().numpy()
        sort_idx_c = np.argsort(xc)
        plt.scatter(xc[sort_idx_c], yc[sort_idx_c], label='Calibration Data', color='green', s=10, alpha=0.5)
        
        c_min, c_max = xc.min(), xc.max()
        plt.axvline(x=c_min, color='green', alpha=0.8, linewidth=0.5, label=f'Calib Min ({c_min:.2f})')
        plt.axvline(x=c_max, color='green', alpha=0.8, linewidth=0.5, label=f'Calib Max ({c_max:.2f})')

    plt.title('Hardware-Exact LUT Simulation Comparison')
    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[VISUALIZATION] Comparison plot saved to: {output_path}")

def to_c_header(tensor, name, output_path):
    """Generates a C header for firmware verification."""
    data = tensor.flatten().detach().cpu().numpy()
    formatted_data = []
    for i in range(0, len(data), 128):
        chunk = data[i : i + 128]
        line = ", ".join([f"{x:.6f}f" for x in chunk])
        formatted_data.append(line)
    
    c_str = ",\n    ".join(formatted_data)
    header_content = (
        "#pragma once\n"
        "// Auto-generated for ESP-DL Firmware Verification\n"
        f"const float {name}[] = {{\n    {c_str}\n}};\n"
        f"const int {name}_len = {len(data)};\n"
    )
    
    with open(output_path, "a" if os.path.exists(output_path) else "w") as f:
        f.write(header_content)
