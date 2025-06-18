import torch
import logging
import collections
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import nodes
import comfy.sd
import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import folder_paths

import comfy.model_management as mm

import os
import torch
import torch.nn.functional as F
import gc
import numpy as np
import math
from tqdm import tqdm

from .wanvideo.modules.t5 import T5EncoderModel
from .wanvideo.modules.model import WanModel

from .ops import GGMLOps, move_patch_to_device
from .loader import gguf_sd_loader, gguf_clip_loader
from .dequant import is_quantized, is_torch_compatible

import logging
from comfy.utils import load_torch_file
script_directory = os.path.dirname(os.path.abspath(__file__))

def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])


class WanVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v

try:
    from comfy.latent_formats import Wan21
    latent_format = Wan21
except: #for backwards compatibility
    logging.warning("Wan21 latent format not found, update ComfyUI for better livepreview")
    from comfy.latent_formats import HunyuanVideo
    latent_format = HunyuanVideo

class WanVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = latent_format
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.0
        self.unet_config["disable_unet_model_creation"] = True

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    mmap_released = False
    def load(self, *args, force_patch_weights=False, **kwargs):
        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.model.named_modules():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked and self.load_device != self.offload_device:
                logging.info(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        if src_cls != GGUFModelPatcher:
            n.size = 0 # force recalc
        return n




class UnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
            },
            "optional": {
                "load_device": (["main_device", "offload_device"], {
                    "default": "offload_device", 
                    "tooltip": "Device to load model to. Use offload_device to save VRAM"
                }),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
                "dequant_dtype": (["default", "target", "fp16", "bf16", "fp32"], {
                    "default": "default", 
                    "tooltip": "Dtype for dequantizing GGUF tensors"
                }),
                "patch_dtype": (["default", "target", "fp16", "bf16", "fp32"], {
                    "default": "default",
                    "tooltip": "Dtype for patch calculations"
                }),
                "patch_on_device": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to keep patches on GPU"
                }),
                "fantasytalking_model": ("FANTASYTALKINGMODEL", {
                    "default": None, 
                    "tooltip": "FantasyTalking model https://github.com/Fantasy-AMAP"
                }),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "load_unet"
    CATEGORY = "bootleg-alt"
    TITLE = "Unet Loader (GGUF)"

    def load_unet(self, unet_name, load_device="offload_device", base_precision="fp16", dequant_dtype="default", patch_dtype="default", patch_on_device=False, fantasytalking_model=None):
        ops = GGMLOps()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            # Map string names to actual torch dtypes
            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16, 
                "fp32": torch.float32
            }
            ops.Linear.dequant_dtype = dtype_map.get(dequant_dtype, torch.float16)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            # Map string names to actual torch dtypes
            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32
            }
            ops.Linear.patch_dtype = dtype_map.get(patch_dtype, torch.float16)

        # init model
        unet_path = folder_paths.get_full_path("unet", unet_name)
        sd = gguf_sd_loader(unet_path)
        
        # Get model dimension from state dict (similar to original WanVideo code)
        model_dim = None
        if "patch_embedding.weight" in sd:
            model_dim = sd["patch_embedding.weight"].shape[0]
        elif "blocks.0.norm1.weight" in sd:
            model_dim = sd["blocks.0.norm1.weight"].shape[0]
        elif "blocks.0.attn.norm.weight" in sd:
            model_dim = sd["blocks.0.attn.norm.weight"].shape[0]

        dim = sd["patch_embedding.weight"].shape[0]
        ffn_dim = sd["blocks.0.ffn.0.bias"].shape[0]

        in_channels = sd["patch_embedding.weight"].shape[1]
        model_type = "i2v"
        num_heads = 40
        num_layers = 40
        attention_mode = "sdpa"
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        base_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[base_precision]
        
        # Choose loading device based on user preference
        transformer_load_device = device if load_device == "main_device" else offload_device

        TRANSFORMER_CONFIG = {
            "dim": dim,
            "ffn_dim": ffn_dim,
            "eps": 1e-06,
            "freq_dim": 256,
            "in_dim": in_channels,
            "model_type": model_type,
            "out_dim": 16,
            "text_len": 512,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "attention_mode": attention_mode,
            "main_device": device,
            "offload_device": offload_device,
            "teacache_coefficients": [],
            "magcache_ratios": [],

            "vace_in_dim": None,
            "inject_sample_info": True if "fps_embedding.weight" in sd else False,
            "add_ref_conv": True if "ref_conv.weight" in sd else False,
            "in_dim_ref_conv": sd["ref_conv.weight"].shape[1] if "ref_conv.weight" in sd else None,
            "add_control_adapter": True if "control_adapter.conv.weight" in sd else False,
        }

        with init_empty_weights():
            transformer = WanModel(**TRANSFORMER_CONFIG)
        transformer.eval()
        
        # FantasyTalking integration
        if fantasytalking_model is not None:
            logging.info("FantasyTalking model detected, patching model...")
            context_dim = fantasytalking_model["sd"]["proj_model.proj.weight"].shape[0]
            import torch.nn as nn
            for block in transformer.blocks:
                block.cross_attn.k_proj = nn.Linear(context_dim, dim, bias=False)
                block.cross_attn.v_proj = nn.Linear(context_dim, dim, bias=False)
            sd.update(fantasytalking_model["sd"])

        comfy_model = WanVideoModel(
            WanVideoModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )

        params_to_keep = {"norm", "head", "bias", "time_in", "vector_in", "patch_embedding", "time_", "img_emb", "modulation", "text_embedding", "adapter", "add"}

    
        logging.info("Using accelerate to load and assign model weights to device...")
        param_count = sum(1 for _ in transformer.named_parameters())
        for name, param in tqdm(transformer.named_parameters(), 
                desc=f"Loading transformer parameters to {transformer_load_device}", 
                total=param_count,
                leave=True):
            dtype_to_use = base_dtype
            if "patch_embedding" in name:
                dtype_to_use = torch.float32
            
            # Handle GGUF quantized tensors properly
            if name in sd:
                tensor_value = sd[name]
                
                # Check if this is a quantized tensor (GGMLTensor)
                if hasattr(tensor_value, 'tensor_type') and is_quantized(tensor_value):
                    # For quantized tensors, we need to set requires_grad=False
                    # Move to device and set as parameter directly
                    tensor_on_device = tensor_value.to(transformer_load_device)
                    param_tensor = torch.nn.Parameter(tensor_on_device, requires_grad=False)
                    comfy.utils.set_attr_param(transformer, name, param_tensor)
                else:
                    # For regular tensors, use the original method
                    set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=tensor_value)
            else:
                # If tensor not in state dict, initialize with empty tensor
                empty_tensor = torch.empty(param.shape, dtype=dtype_to_use, device=transformer_load_device)
                param_tensor = torch.nn.Parameter(empty_tensor, requires_grad=False)
                comfy.utils.set_attr_param(transformer, name, param_tensor)

        
        comfy_model.diffusion_model = transformer
        comfy_model.load_device = transformer_load_device

        # Move to offload device if requested to save VRAM
        if load_device == "offload_device" and transformer.device != offload_device:
            logging.info(f"Moving transformer from {transformer.device} to {offload_device}")
            transformer.to(offload_device)
            gc.collect()
            mm.soft_empty_cache()

        patcher = GGUFModelPatcher(comfy_model, device, offload_device)
        patcher.patch_on_device = patch_on_device
        patcher.model.is_patched = False

        patcher.model["dtype"] = base_dtype
        patcher.model["base_path"] = unet_path
        patcher.model["model_name"] = unet_name
        patcher.model["manual_offloading"] = True
        patcher.model["quantization"] = "disabled"
        patcher.model["auto_cpu_offload"] = False
        patcher.model["control_lora"] = False

        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}
        patcher.model_options["transformer_options"]["block_swap_args"] = None   

        for model in mm.current_loaded_models:
            if model._model() == patcher:
                mm.current_loaded_models.remove(model)            

        return (patcher,)


class UnetLoaderGGUF_LowVRAM:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
            },
            "optional": {
                "fantasytalking_model": ("FANTASYTALKINGMODEL", {
                    "default": None, 
                    "tooltip": "FantasyTalking model https://github.com/Fantasy-AMAP"
                }),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "load_unet"
    CATEGORY = "bootleg-alt"
    TITLE = "Unet Loader (GGUF) - Low VRAM"

    def load_unet(self, unet_name, fantasytalking_model=None):
        """Memory-optimized loader for low VRAM systems"""
        return UnetLoaderGGUF().load_unet(
            unet_name=unet_name,
            load_device="offload_device",  # Always use CPU offload
            base_precision="fp16",         # Use fp16 for memory savings
            dequant_dtype="fp16",         # Dequantize to fp16
            patch_dtype="fp16",           # Use fp16 for patches
            patch_on_device=False,        # Keep patches on CPU
            fantasytalking_model=fantasytalking_model
        )



class CLIPLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.CLIPLoader.INPUT_TYPES()
        return {
            "required": {
                "clip_name": (s.get_filename_list(),),
                "type": base["required"]["type"],
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "bootleg-alt"
    TITLE = "CLIPLoader (GGUF) V2"

    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    def load_data(self, ckpt_paths):
        clip_data = []
        for p in ckpt_paths:
            if p.endswith(".gguf"):
                sd = gguf_clip_loader(p)
            else:
                sd = comfy.utils.load_torch_file(p, safe_load=True)
                if "scaled_fp8" in sd: # NOTE: Scaled FP8 would require different custom ops, but only one can be active
                    raise NotImplementedError(f"Mixing scaled FP8 with GGUF is not supported! Use regular CLIP loader or switch model(s)\n({p})")
            clip_data.append(sd)
        return clip_data

    def load_patcher(self, clip_paths, clip_type, clip_data):
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type = clip_type,
            state_dicts = clip_data,
            model_options = {
                "custom_operations": GGMLOps,
                "initial_device": comfy.model_management.text_encoder_offload_device()
            },
            embedding_directory = folder_paths.get_folder_paths("embeddings"),
        )
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
        return clip

    def load_clip(self, clip_name, type="stable_diffusion"):
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher([clip_path], clip_type, self.load_data([clip_path])),)


class LoadWanVideoT5TextEncoderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (s.get_filename_list(), {"tooltip": "These models are loaded from 'ComfyUI/models/text_encoders' and 'ComfyUI/models/clip_gguf'"}),
                "precision": (["fp32", "bf16"],
                    {"default": "bf16"}
                ),
            },
            "optional": {
                "load_device": (["main_device", "offload_device"], {"default": "offload_device"}),
                "quantization": (['disabled', 'fp8_e4m3fn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            }
        }

    RETURN_TYPES = ("WANTEXTENCODER",)
    RETURN_NAMES = ("wan_t5_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "bootleg-alt"
    TITLE = "LoadWanVideoT5TextEncoder (GGUF)"
    DESCRIPTION = "Loads Wan T5 text_encoder model from 'ComfyUI/models/text_encoders' or 'ComfyUI/models/clip_gguf' (supports GGUF)"

    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list("text_encoders")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    def loadmodel(self, model_name, precision, load_device="offload_device", quantization="disabled"):
       
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        text_encoder_load_device = device if load_device == "main_device" else offload_device

        tokenizer_path = os.path.join(script_directory, "configs", "T5_tokenizer")

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        # Try to get full path from both folders
        model_path = None
        try:
            model_path = folder_paths.get_full_path("text_encoders", model_name)
        except:
            try:
                model_path = folder_paths.get_full_path("clip_gguf", model_name)
            except:
                raise ValueError(f"Model {model_name} not found in text_encoders or clip_gguf folders")

        # Load model data - handle both GGUF and regular formats
        if model_path.endswith(".gguf"):
            sd = gguf_clip_loader(model_path)
        else:
            sd = load_torch_file(model_path, safe_load=True)
        
        if "token_embedding.weight" not in sd and "shared.weight" not in sd:
            raise ValueError("Invalid T5 text encoder model, this node expects the 'umt5-xxl' model")
        if "scaled_fp8" in sd:
            raise ValueError("Invalid T5 text encoder model, fp8 scaled is not supported by this node")

        # Convert state dict keys from T5 format to the expected format
        if "shared.weight" in sd:
            logging.info("Converting T5 text encoder model to the expected format...")
            converted_sd = {}
            
            for key, value in sd.items():
                # Handle encoder block patterns
                if key.startswith('encoder.block.'):
                    parts = key.split('.')
                    block_num = parts[2]
                    
                    # Self-attention components
                    if 'layer.0.SelfAttention' in key:
                        if key.endswith('.k.weight'):
                            new_key = f"blocks.{block_num}.attn.k.weight"
                        elif key.endswith('.o.weight'):
                            new_key = f"blocks.{block_num}.attn.o.weight"
                        elif key.endswith('.q.weight'):
                            new_key = f"blocks.{block_num}.attn.q.weight"
                        elif key.endswith('.v.weight'):
                            new_key = f"blocks.{block_num}.attn.v.weight"
                        elif 'relative_attention_bias' in key:
                            new_key = f"blocks.{block_num}.pos_embedding.embedding.weight"
                        else:
                            new_key = key
                    
                    # Layer norms
                    elif 'layer.0.layer_norm' in key:
                        new_key = f"blocks.{block_num}.norm1.weight"
                    elif 'layer.1.layer_norm' in key:
                        new_key = f"blocks.{block_num}.norm2.weight"
                    
                    # Feed-forward components
                    elif 'layer.1.DenseReluDense' in key:
                        if 'wi_0' in key:
                            new_key = f"blocks.{block_num}.ffn.gate.0.weight"
                        elif 'wi_1' in key:
                            new_key = f"blocks.{block_num}.ffn.fc1.weight"
                        elif 'wo' in key:
                            new_key = f"blocks.{block_num}.ffn.fc2.weight"
                        else:
                            new_key = key
                    else:
                        new_key = key
                elif key == "shared.weight":
                    new_key = "token_embedding.weight"
                elif key == "encoder.final_layer_norm.weight":
                    new_key = "norm.weight"
                else:
                    new_key = key
                converted_sd[new_key] = value
            sd = converted_sd

        T5_text_encoder = T5EncoderModel(
            text_len=512,
            dtype=dtype,
            device=text_encoder_load_device,
            state_dict=sd,
            tokenizer_path=tokenizer_path,
            quantization=quantization
        )
        text_encoder = {
            "model": T5_text_encoder,
            "dtype": dtype,
        }
        
        return (text_encoder,)

NODE_CLASS_MAPPINGS = {
    "UnetLoaderGGUF": UnetLoaderGGUF,
    "UnetLoaderGGUF_LowVRAM": UnetLoaderGGUF_LowVRAM,
    "CLIPLoaderGGUF": CLIPLoaderGGUF,
    "LoadWanVideoT5TextEncoderGGUF": LoadWanVideoT5TextEncoderGGUF,
}

"""
!!! Exception during processing !!! mat1 and mat2 shapes cannot be multiplied (512x4096 and 3360x4096)
Traceback (most recent call last):
  File "C:\Users\Rahel\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\execution.py", line 361, in execute
    output_data, output_ui, has_subgraph = get_output_data(obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\execution.py", line 236, in get_output_data
    return_values = _map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\execution.py", line 208, in _map_node_over_list
    process_inputs(input_dict, i)
  File "C:\Users\Rahel\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\execution.py", line 197, in process_inputs
    results.append(getattr(obj, func)(**inputs))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\nodes.py", line 1233, in process
    context = encoder(positive_prompts, device)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\custom_nodes\ComfyUI-GGUF-FantasyTalking\wanvideo\modules\t5.py", line 518, in __call__
    context = self.model(ids, mask)
              ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\custom_nodes\ComfyUI-GGUF-FantasyTalking\wanvideo\modules\t5.py", line 311, in forward
    x = block(x, mask, pos_bias=e)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\custom_nodes\ComfyUI-GGUF-FantasyTalking\wanvideo\modules\t5.py", line 175, in forward
    x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\custom_nodes\ComfyUI-GGUF-FantasyTalking\wanvideo\modules\t5.py", line 99, in forward
    q = self.q(x).view(b, -1, n, c)
        ^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Rahel\Documents\ComfyUI\.venv\Lib\site-packages\torch\_tensor.py", line 1668, in __torch_function__
    ret = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (512x4096 and 3360x4096)

Prompt executed in 88.52 seconds"""