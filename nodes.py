import torch
import logging
import collections

import nodes
import comfy.sd
import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import folder_paths

from .ops import GGMLOps, move_patch_to_device
from .loader import gguf_sd_loader, gguf_clip_loader
from .dequant import is_quantized, is_torch_compatible

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
                "fantasytalking_model": ("FANTASYTALKINGMODEL", {
                    "default": None, 
                    "tooltip": "FantasyTalking model https://github.com/Fantasy-AMAP"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "bootleg"
    TITLE = "Unet Loader (GGUF)"

    def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None, fantasytalking_model=None):
        ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        # init model
        unet_path = folder_paths.get_full_path("unet", unet_name)
        sd = gguf_sd_loader(unet_path)
        
        # FantasyTalking integration
        if fantasytalking_model is not None:
            logging.info("FantasyTalking model detected, merging state dict...")
            fantasy_sd = fantasytalking_model["sd"]
            
            # Merge FantasyTalking weights into the main state dict
            sd.update(fantasy_sd)
            
            # Store context dimension for potential model patching
            context_dim = fantasy_sd["proj_model.proj.weight"].shape[0]
            logging.info(f"FantasyTalking context dimension: {context_dim}")

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}
        )
        
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        
        # Apply FantasyTalking model patches if needed
        if fantasytalking_model is not None:
            model = self._apply_fantasytalking_patches(model, fantasytalking_model)
        
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return (model,)
    
    def _apply_fantasytalking_patches(self, model, fantasytalking_model):
        """Apply FantasyTalking specific patches to the model"""
        import torch.nn as nn
        
        fantasy_sd = fantasytalking_model["sd"]
        context_dim = fantasy_sd["proj_model.proj.weight"].shape[0]
        
        # Check if model has transformer blocks structure
        if hasattr(model.model, 'diffusion_model') and hasattr(model.model.diffusion_model, 'blocks'):
            diffusion_model = model.model.diffusion_model
            dim = diffusion_model.blocks[0].cross_attn.q_proj.in_features if hasattr(diffusion_model.blocks[0], 'cross_attn') else None
            
            if dim is not None:
                logging.info(f"Patching cross-attention layers for FantasyTalking (dim: {dim}, context_dim: {context_dim})")
                
                # Patch each transformer block's cross-attention
                for block in diffusion_model.blocks:
                    if hasattr(block, 'cross_attn'):
                        # Replace k_proj and v_proj to match context dimension
                        block.cross_attn.k_proj = nn.Linear(context_dim, dim, bias=False)
                        block.cross_attn.v_proj = nn.Linear(context_dim, dim, bias=False)
                        
                        # Move to same device and dtype as original model
                        device = next(block.parameters()).device
                        dtype = next(block.parameters()).dtype
                        block.cross_attn.k_proj.to(device=device, dtype=dtype)
                        block.cross_attn.v_proj.to(device=device, dtype=dtype)
            else:
                logging.warning("Could not determine model dimension for FantasyTalking patching")
        else:
            # For models without transformer block structure, try alternative patching
            logging.info("Attempting alternative FantasyTalking patching for non-transformer model")
            self._apply_alternative_fantasytalking_patches(model, fantasytalking_model)
        
        return model
    
    def _apply_alternative_fantasytalking_patches(self, model, fantasytalking_model):
        """Alternative patching method for models with different architectures"""
        import torch.nn as nn
        
        fantasy_sd = fantasytalking_model["sd"]
        context_dim = fantasy_sd["proj_model.proj.weight"].shape[0]
        
        # Search for cross-attention modules in the model recursively
        def patch_cross_attention_recursive(module, context_dim):
            for name, child in module.named_children():
                if 'cross_attn' in name.lower() or 'crossattn' in name.lower():
                    # Found cross-attention module, try to patch it
                    if hasattr(child, 'k_proj') and hasattr(child, 'v_proj'):
                        dim = child.k_proj.out_features
                        child.k_proj = nn.Linear(context_dim, dim, bias=False)
                        child.v_proj = nn.Linear(context_dim, dim, bias=False)
                        
                        device = next(child.parameters()).device
                        dtype = next(child.parameters()).dtype
                        child.k_proj.to(device=device, dtype=dtype)
                        child.v_proj.to(device=device, dtype=dtype)
                        
                        logging.info(f"Patched cross-attention module: {name}")
                else:
                    # Recursively search in child modules
                    patch_cross_attention_recursive(child, context_dim)
        
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            patch_cross_attention_recursive(model.model.diffusion_model, context_dim)
        else:
            patch_cross_attention_recursive(model, context_dim)