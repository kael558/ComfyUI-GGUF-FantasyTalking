class WanVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16", "fp16_fast"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2', 'fp8_e4m3fn_fast_no_ffn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn_2",
                    "flash_attn_3",
                    "sageattn",
                    "flex_attention",
                    #"spargeattn", needs tuning
                    #"spargeattn_tune",
                    ], {"default": "sdpa"}),
                "compile_args": ("WANCOMPILEARGS", ),
                "block_swap_args": ("BLOCKSWAPARGS", ),
                "lora": ("WANVIDLORA", {"default": None}),
                "vram_management_args": ("VRAM_MANAGEMENTARGS", {"default": None, "tooltip": "Alternative offloading method from DiffSynth-Studio, more aggressive in reducing memory use than block swapping, but can be slower"}),
                "vace_model": ("VACEPATH", {"default": None, "tooltip": "VACE model to use when not using model that has it included"}),
                "fantasytalking_model": ("FANTASYTALKINGMODEL", {"default": None, "tooltip": "FantasyTalking model https://github.com/Fantasy-AMAP"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision, load_device,  quantization,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None, vram_management_args=None, vace_model=None, fantasytalking_model=None):
        assert not (vram_management_args is not None and block_swap_args is not None), "Can't use both block_swap_args and vram_management_args at the same time"
        lora_low_mem_load = False
        if lora is not None:
            for l in lora:
                lora_low_mem_load = l.get("low_mem_load") if lora is not None else False

        transformer = None
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        manual_offloading = True
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

                
        manual_offloading = True
        transformer_load_device = device if load_device == "main_device" else offload_device
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        
        if base_precision == "fp16_fast":
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                raise ValueError("torch.backends.cuda.matmul.allow_fp16_accumulation is not available in this version of torch, requires torch 2.7.0.dev2025 02 26 nightly minimum currently")
        else:
            try:
                if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                    torch.backends.cuda.matmul.allow_fp16_accumulation = False
            except:
                pass

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
      
        sd = load_torch_file(model_path, device=transformer_load_device, safe_load=True)

        
        if "vace_blocks.0.after_proj.weight" in sd and not "patch_embedding.weight" in sd:
            raise ValueError("You are attempting to load a VACE module as a WanVideo model, instead you should use the vace_model input and matching T2V base model")

        if vace_model is not None:
            vace_sd = load_torch_file(vace_model["path"], device=transformer_load_device, safe_load=True)
            sd.update(vace_sd)

        first_key = next(iter(sd))
        if first_key.startswith("model.diffusion_model."):
            new_sd = {}
            for key, value in sd.items():
                new_key = key.replace("model.diffusion_model.", "", 1)
                new_sd[new_key] = value
            sd = new_sd
        elif first_key.startswith("model."):
            new_sd = {}
            for key, value in sd.items():
                new_key = key.replace("model.", "", 1)
                new_sd[new_key] = value
            sd = new_sd
        if not "patch_embedding.weight" in sd:
            raise ValueError("Invalid WanVideo model selected")
        dim = sd["patch_embedding.weight"].shape[0]
        in_channels = sd["patch_embedding.weight"].shape[1]
        log.info(f"Detected model in_channels: {in_channels}")
        ffn_dim = sd["blocks.0.ffn.0.bias"].shape[0]

        if not "text_embedding.0.weight" in sd:
            model_type = "no_cross_attn" #minimaxremover
        elif "model_type.Wan2_1-FLF2V-14B-720P" in sd or "img_emb.emb_pos" in sd or "flf2v" in model.lower():
            model_type = "fl2v"
        elif in_channels in [36, 48]:
            model_type = "i2v"
        elif in_channels == 16:
            model_type = "t2v"
        elif "control_adapter.conv.weight" in sd:
            model_type = "t2v"

        num_heads = 40 if dim == 5120 else 12
        num_layers = 40 if dim == 5120 else 30


        log.info(f"Model type: {model_type}, num_heads: {num_heads}, num_layers: {num_layers}")

        teacache_coefficients_map = {
            "1_3B": {
                "e": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
                "e0": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            },
            "14B": {
                "e": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
                "e0": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            },
            "i2v_480": {
                "e": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
                "e0": [2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            },
            "i2v_720":{
                "e": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
                "e0": [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02],
            },
        }

        magcache_ratios_map = {
            "1_3B": np.array([1.0]*2+[1.0124, 1.02213, 1.00166, 1.0041, 0.99791, 1.00061, 0.99682, 0.99762, 0.99634, 0.99685, 0.99567, 0.99586, 0.99416, 0.99422, 0.99578, 0.99575, 0.9957, 0.99563, 0.99511, 0.99506, 0.99535, 0.99531, 0.99552, 0.99549, 0.99541, 0.99539, 0.9954, 0.99536, 0.99489, 0.99485, 0.99518, 0.99514, 0.99484, 0.99478, 0.99481, 0.99479, 0.99415, 0.99413, 0.99419, 0.99416, 0.99396, 0.99393, 0.99388, 0.99386, 0.99349, 0.99349, 0.99309, 0.99304, 0.9927, 0.9927, 0.99228, 0.99226, 0.99171, 0.9917, 0.99137, 0.99135, 0.99068, 0.99063, 0.99005, 0.99003, 0.98944, 0.98942, 0.98849, 0.98849, 0.98758, 0.98757, 0.98644, 0.98643, 0.98504, 0.98503, 0.9836, 0.98359, 0.98202, 0.98201, 0.97977, 0.97978, 0.97717, 0.97718, 0.9741, 0.97411, 0.97003, 0.97002, 0.96538, 0.96541, 0.9593, 0.95933, 0.95086, 0.95089, 0.94013, 0.94019, 0.92402, 0.92414, 0.90241, 0.9026, 0.86821, 0.86868, 0.81838, 0.81939]),
            "14B": np.array([1.0]*2+[1.02504, 1.03017, 1.00025, 1.00251, 0.9985, 0.99962, 0.99779, 0.99771, 0.9966, 0.99658, 0.99482, 0.99476, 0.99467, 0.99451, 0.99664, 0.99656, 0.99434, 0.99431, 0.99533, 0.99545, 0.99468, 0.99465, 0.99438, 0.99434, 0.99516, 0.99517, 0.99384, 0.9938, 0.99404, 0.99401, 0.99517, 0.99516, 0.99409, 0.99408, 0.99428, 0.99426, 0.99347, 0.99343, 0.99418, 0.99416, 0.99271, 0.99269, 0.99313, 0.99311, 0.99215, 0.99215, 0.99218, 0.99215, 0.99216, 0.99217, 0.99163, 0.99161, 0.99138, 0.99135, 0.98982, 0.9898, 0.98996, 0.98995, 0.9887, 0.98866, 0.98772, 0.9877, 0.98767, 0.98765, 0.98573, 0.9857, 0.98501, 0.98498, 0.9838, 0.98376, 0.98177, 0.98173, 0.98037, 0.98035, 0.97678, 0.97677, 0.97546, 0.97543, 0.97184, 0.97183, 0.96711, 0.96708, 0.96349, 0.96345, 0.95629, 0.95625, 0.94926, 0.94929, 0.93964, 0.93961, 0.92511, 0.92504, 0.90693, 0.90678, 0.8796, 0.87945, 0.86111, 0.86189]),
            "i2v_480": np.array([1.0]*2+[0.98783, 0.98993, 0.97559, 0.97593, 0.98311, 0.98319, 0.98202, 0.98225, 0.9888, 0.98878, 0.98762, 0.98759, 0.98957, 0.98971, 0.99052, 0.99043, 0.99383, 0.99384, 0.98857, 0.9886, 0.99065, 0.99068, 0.98845, 0.98847, 0.99057, 0.99057, 0.98957, 0.98961, 0.98601, 0.9861, 0.98823, 0.98823, 0.98756, 0.98759, 0.98808, 0.98814, 0.98721, 0.98724, 0.98571, 0.98572, 0.98543, 0.98544, 0.98157, 0.98165, 0.98411, 0.98413, 0.97952, 0.97953, 0.98149, 0.9815, 0.9774, 0.97742, 0.97825, 0.97826, 0.97355, 0.97361, 0.97085, 0.97087, 0.97056, 0.97055, 0.96588, 0.96587, 0.96113, 0.96124, 0.9567, 0.95681, 0.94961, 0.94969, 0.93973, 0.93988, 0.93217, 0.93224, 0.91878, 0.91896, 0.90955, 0.90954, 0.92617, 0.92616]),
            "i2v_720": np.array([1.0]*2+[0.99428, 0.99498, 0.98588, 0.98621, 0.98273, 0.98281, 0.99018, 0.99023, 0.98911, 0.98917, 0.98646, 0.98652, 0.99454, 0.99456, 0.9891, 0.98909, 0.99124, 0.99127, 0.99102, 0.99103, 0.99215, 0.99212, 0.99515, 0.99515, 0.99576, 0.99572, 0.99068, 0.99072, 0.99097, 0.99097, 0.99166, 0.99169, 0.99041, 0.99042, 0.99201, 0.99198, 0.99101, 0.99101, 0.98599, 0.98603, 0.98845, 0.98844, 0.98848, 0.98851, 0.98862, 0.98857, 0.98718, 0.98719, 0.98497, 0.98497, 0.98264, 0.98263, 0.98389, 0.98393, 0.97938, 0.9794, 0.97535, 0.97536, 0.97498, 0.97499, 0.973, 0.97301, 0.96827, 0.96828, 0.96261, 0.96263, 0.95335, 0.9534, 0.94649, 0.94655, 0.93397, 0.93414, 0.91636, 0.9165, 0.89088, 0.89109, 0.8679, 0.86768]),
        }

        model_variant = "14B" #default to this
        if model_type == "i2v" or model_type == "fl2v":
            if "480" in model or "fun" in model.lower() or "a2" in model.lower() or "540" in model: #just a guess for the Fun model for now...
                model_variant = "i2v_480"
            elif "720" in model:
                model_variant = "i2v_720"
        elif model_type == "t2v":
            model_variant = "14B"
            
        if dim == 1536:
            model_variant = "1_3B"
        log.info(f"Model variant detected: {model_variant}")
        
        TRANSFORMER_CONFIG= {
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
            "teacache_coefficients": teacache_coefficients_map[model_variant],
            "magcache_ratios": magcache_ratios_map[model_variant],

            "vace_in_dim": vace_in_dim,
            "inject_sample_info": True if "fps_embedding.weight" in sd else False,
            "add_ref_conv": True if "ref_conv.weight" in sd else False,
            "in_dim_ref_conv": sd["ref_conv.weight"].shape[1] if "ref_conv.weight" in sd else None,
            "add_control_adapter": True if "control_adapter.conv.weight" in sd else False,
        }

        with init_empty_weights():
            transformer = WanModel(**TRANSFORMER_CONFIG)
        transformer.eval()

        # FantasyTalking https://github.com/Fantasy-AMAP
        if fantasytalking_model is not None:
            log.info("FantasyTalking model detected, patching model...")
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
        
        if quantization == "disabled":
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.float8_e4m3fn:
                        quantization = "fp8_e4m3fn"
                        break
                    elif v.dtype == torch.float8_e5m2:
                        quantization = "fp8_e5m2"
                        break

        if "fp8_e4m3fn" in quantization:
            dtype = torch.float8_e4m3fn
        elif quantization == "fp8_e5m2":
            dtype = torch.float8_e5m2
        else:
            dtype = base_dtype
        params_to_keep = {"norm", "head", "bias", "time_in", "vector_in", "patch_embedding", "time_", "img_emb", "modulation", "text_embedding", "adapter", "add"}
        #if lora is not None:
        #    transformer_load_device = device
        if not lora_low_mem_load:
            log.info("Using accelerate to load and assign model weights to device...")
            param_count = sum(1 for _ in transformer.named_parameters())
            for name, param in tqdm(transformer.named_parameters(), 
                    desc=f"Loading transformer parameters to {transformer_load_device}", 
                    total=param_count,
                    leave=True):
                dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                if "patch_embedding" in name:
                    dtype_to_use = torch.float32
                set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])
        comfy_model.diffusion_model = transformer
        comfy_model.load_device = transformer_load_device
        
        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)
        patcher.model.is_patched = False

        control_lora = False
        
        #compile
        if compile_args is not None and vram_management_args is None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            try:
                if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
                    torch._dynamo.config.recompile_limit = compile_args["dynamo_recompile_limit"]
            except Exception as e:
                log.warning(f"Could not set recompile_limit: {e}")
            if compile_args["compile_transformer_blocks_only"]:
                for i, block in enumerate(patcher.model.diffusion_model.blocks):
                    patcher.model.diffusion_model.blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

            else:
                patcher.model.diffusion_model = torch.compile(patcher.model.diffusion_model, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])        
        
        if load_device == "offload_device" and patcher.model.diffusion_model.device != offload_device:
            log.info(f"Moving diffusion model from {patcher.model.diffusion_model.device} to {offload_device}")
            patcher.model.diffusion_model.to(offload_device)
            gc.collect()
            mm.soft_empty_cache()

        patcher.model["dtype"] = base_dtype
        patcher.model["base_path"] = model_path
        patcher.model["model_name"] = model
        patcher.model["manual_offloading"] = manual_offloading
        patcher.model["quantization"] = quantization
        patcher.model["auto_cpu_offload"] = True if vram_management_args is not None else False
        patcher.model["control_lora"] = control_lora

        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}
        patcher.model_options["transformer_options"]["block_swap_args"] = block_swap_args   

        for model in mm.current_loaded_models:
            if model._model() == patcher:
                mm.current_loaded_models.remove(model)            

        return (patcher,)