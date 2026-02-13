# ESP32-P4 Firmware Verification

This project is used to verify the bit-exactness of the Swish LUT implementation on real hardware.

## ⚙️ Configuration Reminder

Before building, you **must** ensure the `esp-dl` component path is correctly set for your local environment.

1.  Open `main/idf_component.yml`.
2.  Update the `path` under `esp-dl` to point to your local clone of the `esp-dl` repository.

```yaml
dependencies:
  esp-dl:
    path: /absolute/path/to/your/esp-dl
```

After updating the path, run `idf.py build` to verify.
