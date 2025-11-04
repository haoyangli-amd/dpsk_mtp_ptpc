
from safetensors.torch import load_file

file_path1 = "/mnt/raid0/pretrained_model/ptpc_mtp/DeepSeek-R1-BF16-FP8-ptpc-mtp/model-00135-of-00139.safetensors"
file_path2 = "/mnt/raid0/pretrained_model/unsloth/DeepSeek-R1-BF16/model-00160-of-000163.safetensors"
state_dict1 = load_file(file_path1)
state_dict2 = load_file(file_path2)
for key, tensor in state_dict1.items():
    pass


