# quantize deepseekr1 with mtp by amd-quark (PTPC)

## Installation

To get started, install:

```bash
pip install amd-quark
```

## Quickstart

### 1) Applying changes to transformers locally (using transformers==4.54.0)

```bash
diff modeling_deepseek_v3.py modeling_deepseek_v3_modified.py
```

### 2) Execute the following script

Please change the file path in the script

```bash
python3 dpsk_ptpc_mtp.py
```

### 3) Check that the quantization is as expected by checking the safetensor

Please change the file path in the script

```bash
python3 read_safe.py
```

