# ComfyUI-GGUF-FantasyTalking OOM Error Fixes

## Quick Fix for Out of Memory Errors

### Option 1: Use the Low VRAM Loader (Easiest)

1. In ComfyUI, use the **"Unet Loader (GGUF) - Low VRAM"** node instead of the regular one
2. This automatically uses the most memory-efficient settings:
   - Loads model to CPU (offload_device)
   - Uses FP16 precision
   - Keeps patches on CPU
   - Optimized memory cleanup

### Option 2: Configure the Regular Loader

If using the regular **"Unet Loader (GGUF)"** node, use these settings:

- **load_device**: `offload_device` (loads to CPU to save VRAM)
- **base_precision**: `fp16` (uses less memory than fp32)
- **dequant_dtype**: `fp16` (dequantizes to fp16 instead of fp32)
- **patch_dtype**: `fp16` (uses fp16 for LoRA patches)
- **patch_on_device**: `False` (keeps patches on CPU)

### Additional Memory Optimization Tips

1. **Close other applications** that use VRAM
2. **Restart ComfyUI** before loading large models
3. **Use smaller batch sizes** in your workflow
4. **Enable model offloading** in ComfyUI settings
5. **Consider upgrading VRAM** if regularly hitting limits

### Technical Details

The OOM errors were caused by:

- Models loading directly to GPU memory
- Using FP32 precision unnecessarily
- Inefficient memory management during loading
- Lack of device selection options

These fixes:

- Add CPU offloading options
- Implement proper memory cleanup
- Provide FP16 precision options
- Add device selection controls
- Include low VRAM preset

### Troubleshooting

If you still get OOM errors:

1. Try the Low VRAM loader first
2. Reduce your video resolution/length
3. Close other GPU applications
4. Restart ComfyUI completely
5. Check available VRAM with `nvidia-smi`
