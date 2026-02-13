import subprocess
import os

def run_test(script_path):
    print(f"\n>>> Executing: {script_path}")
    result = subprocess.run(["python", script_path], capture_output=False)
    if result.returncode == 0:
        print(f">>> {script_path} Completed successfully.")
    else:
        print(f">>> {script_path} Failed with exit code {result.returncode}.")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    tests = [
        os.path.join(SCRIPT_DIR, "test_swish.py"),
        os.path.join(SCRIPT_DIR, "test_sigmoid.py"),
        os.path.join(SCRIPT_DIR, "test_tanh.py")
    ]
    
    for test in tests:
        if os.path.exists(test):
            run_test(test)
        else:
            print(f"Error: Script not found at {test}")

    print("\n" + "\033[93m" + "="*80 + "\033[0m")
    print(" " * 25 + "\033[1;92mALL TESTS GENERATED\033[0m")
    print("\033[93m" + "="*80 + "\033[0m")
    print("""
\033[1;36mWHAT TO DO NEXT:\033[0m
----------------
1. \033[94mNavigate to the firmware directory:\033[0m
   cd validate_lut_exact_3/tests_3_layers/firmware

2. \033[94mSet the ESP-IDF target:\033[0m
   idf.py set-target [esp32p4 | esp32s3]

3. \033[94mBuild, Flash, and Monitor the verification suite:\033[0m
   idf.py build flash monitor

\033[1;36mWHAT TO EXPECT:\033[0m
---------------
- The firmware will automatically loop through Swish, Sigmoid, and Tanh.
- It loads the .espdl models from 'main/models/' and compares them against 
  the golden headers in 'main/test_data/'.
- LOOK FOR: "\033[32m✅ SUCCESS: [LayerName] matches 100% (Bit-Exact)\033[0m"
- This confirms that your Python simulation perfectly predicts Hardware behavior.

\033[90mEXAMPLE LOGS:\033[0m
I (1484) LUT_VALIDATION: ----------------------------------------------------------
I (1514) LUT_VALIDATION: Testing Layer: Swish
I (1564) LUT_VALIDATION:   \033[32m✅ SUCCESS: Swish matches 100% (Bit-Exact)\033[0m
I (1564) LUT_VALIDATION: ----------------------------------------------------------
I (1574) LUT_VALIDATION: Testing Layer: Sigmoid
I (1624) LUT_VALIDATION:   \033[32m✅ SUCCESS: Sigmoid matches 100% (Bit-Exact)\033[0m
...
I (1674) LUT_VALIDATION: Validation Suite Finished.
""")
    print("\033[93m" + "="*80 + "\033[0m\n")
