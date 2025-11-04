
#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from transformers import AutoTokenizer, AutoModelForCausalLM

from quark.torch import ModelQuantizer, export_safetensors
from quark.torch.quantization import FP8E4M3PerChannelSpec
from quark.torch.quantization.config.config import Config, QuantizationConfig

# Load the original floating-point model
ckpt_path = "/models/unsloth/DeepSeek-R1-BF16"
model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="auto", torch_dtype="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

# Set the quantization configuration
FP8_PER_CHANNEL_SPEC = FP8E4M3PerChannelSpec(is_dynamic=False, ch_axis=0).to_quantization_spec()

FP8_PER_TOKEN_DYNAMIC_SPEC = FP8E4M3PerChannelSpec(is_dynamic=True, ch_axis=1).to_quantization_spec()


W_FP8_PER_CHANNEL_STATIC_A_FP8_PER_TOKEN_DYNAMIC_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TOKEN_DYNAMIC_SPEC,
                                                                             weight=FP8_PER_CHANNEL_SPEC)

quant_config = Config(global_quant_config=W_FP8_PER_CHANNEL_STATIC_A_FP8_PER_TOKEN_DYNAMIC_CONFIG, exclude=["lm_head", "*mlp.gate", "model.layers.61.eh_proj", "model.layers.61.shared_head.head"])

# Apply quantization
quantizer = ModelQuantizer(quant_config, multi_device=True)
model = quantizer.quantize_model(model)

# Export quantized model
output_dir = ckpt_path.rstrip("/").split("/")[-1] + "-FP8-ptpc"
model = quantizer.freeze(model)
export_safetensors(model, output_dir)
tokenizer.save_pretrained(output_dir)
