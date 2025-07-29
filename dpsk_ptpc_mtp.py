from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "/mnt/raid0/pretrained_model/unsloth/DeepSeek-R1-BF16"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the simple PTQ quantization
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head", "re:.*mlp.gate$", "model.layers.61.eh_proj", "model.layers.61.shared_head.head"])

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Save the model.
SAVE_DIR = "/mnt/raid0/pretrained_model/ptpc_mtp/" + MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-ptpc-mtp"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# MODEL_ID = "/mnt/raid0/pretrained_model/Qwen/Qwen2.5-1.5B-Instruct"
# recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head", "re:.*gate_proj$"])
# recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head", "model.layers.0.mlp.gate_proj"])