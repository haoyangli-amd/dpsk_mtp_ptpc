
from safetensors.torch import load_file
import torch
if 0:
    file_path1 = "/mnt/raid0/pretrained_model/ptpc_mtp/DeepSeek-R1-BF16-FP8-ptpc-mtp/model-00138-of-00139.safetensors"
    file_path2 = "/mnt/raid0/pretrained_model/unsloth/DeepSeek-R1-BF16/model-00163-of-000163.safetensors"
    state_dict1 = load_file(file_path1)
    state_dict2 = load_file(file_path2)
    for key, tensor in state_dict1.items():
        if "shared_head" in key:
            print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
            t1 = state_dict1[key]
            t2 = state_dict2[key]
            breakpoint()
if 0:
    file_path1 = "/mnt/raid0/pretrained_model/ptpc_mtp/DeepSeek-R1-BF16-FP8-ptpc-mtp/model-00137-of-00139.safetensors"
    file_path2 = "/mnt/raid0/pretrained_model/unsloth/DeepSeek-R1-BF16/model-00160-of-000163.safetensors"
    state_dict1 = load_file(file_path1)
    state_dict2 = load_file(file_path2)
    for key, tensor in state_dict1.items():
        if "gate.weight" in key:
            print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
            t1 = state_dict1[key]
            t2 = state_dict2[key]
            breakpoint()

if 0:
    file_path1 = "/mnt/raid0/pretrained_model/ptpc_mtp/DeepSeek-R1-BF16-FP8-ptpc-mtp/model-00138-of-00139.safetensors"
    file_path2 = "/mnt/raid0/pretrained_model/unsloth/DeepSeek-R1-BF16/model-00163-of-000163.safetensors"
    state_dict1 = load_file(file_path1)
    state_dict2 = load_file(file_path2)
    for key, tensor in state_dict1.items():
        if "eh_proj." in key:
            print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
            t1 = state_dict1[key]
            t2 = state_dict2[key]
            breakpoint()
if 1:
    file_path1 = "/mnt/raid0/pretrained_model/ptpc_mtp/DeepSeek-R1-BF16-FP8-ptpc-mtp/model-00135-of-00139.safetensors"
    file_path2 = "/mnt/raid0/pretrained_model/unsloth/DeepSeek-R1-BF16/model-00160-of-000163.safetensors"
    state_dict1 = load_file(file_path1)
    state_dict2 = load_file(file_path2)
    for key, tensor in state_dict1.items():
        if "q_a_proj" in key:
            print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
            t1 = state_dict1[key]
            t2 = state_dict2[key]
            breakpoint()

'''
model.layers.60.mlp.experts.90.up_proj.weight: shape=torch.Size([2048, 7168]), dtype=torch.float8_e4m3fn
model.layers.60.mlp.experts.90.up_proj.weight_scale: shape=torch.Size([2048, 1]), dtype=torch.bfloat16
model.layers.60.self_attn.kv_a_layernorm.weight: shape=torch.Size([512]), dtype=torch.bfloat16
model.layers.60.self_attn.kv_a_proj_with_mqa.weight: shape=torch.Size([576, 7168]), dtype=torch.float8_e4m3fn
model.layers.60.self_attn.kv_a_proj_with_mqa.weight_scale: shape=torch.Size([576, 1]), dtype=torch.bfloat16
model.layers.60.self_attn.kv_b_proj.weight: shape=torch.Size([32768, 512]), dtype=torch.float8_e4m3fn
model.layers.60.self_attn.kv_b_proj.weight_scale: shape=torch.Size([32768, 1]), dtype=torch.bfloat16
model.layers.60.self_attn.o_proj.weight: shape=torch.Size([7168, 16384]), dtype=torch.float8_e4m3fn
model.layers.60.self_attn.o_proj.weight_scale: shape=torch.Size([7168, 1]), dtype=torch.bfloat16
model.layers.60.self_attn.q_a_layernorm.weight: shape=torch.Size([1536]), dtype=torch.bfloat16
model.layers.60.self_attn.q_a_proj.weight: shape=torch.Size([1536, 7168]), dtype=torch.float8_e4m3fn
model.layers.60.self_attn.q_a_proj.weight_scale: shape=torch.Size([1536, 1]), dtype=torch.bfloat16
model.layers.60.self_attn.q_b_proj.weight: shape=torch.Size([24576, 1536]), dtype=torch.float8_e4m3fn
model.layers.60.self_attn.q_b_proj.weight_scale: shape=torch.Size([24576, 1]), dtype=torch.bfloat16


'''