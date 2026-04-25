# Radiance — Changes by Albabit
**Branch:** `ltxv23-support-and-various-fixes`  
**Fork:** https://github.com/Albabit/radiance/tree/ltxv23-support-and-various-fixes  
**Based on:** `fxtdstudios/radiance` v2.3.3  
**Author:** [@Albabit](https://github.com/Albabit)  
**Date:** April 2026

---

## Overview

This branch adds full LTX-Video 2.3 support (model loading, audio-video latent handling, correct 128-channel format), fixes several bugs in the Write node (quality control, audio export, H.264 encoder), improves the Resolution node (new presets, crop output, smart latent calculation), adds temporal decoding to the VAE decode node, and implements a clean SUPIR diffusion upscaler integration in the AI Upscale node.

**Files modified:** `nodes_loader.py`, `nodes_resolution.py`, `nodes_sampler.py`, `nodes_io.py`, `nodes_depth.py`, `hdr/vae.py`, `image/upscale.py`  
**Files modified (JS):** `js/radiance_io.js`, `js/radiance_resolution.js`, `js/radiance_sampler.js`  
**Files added (JS):** `js/radiance_loader.js`, `js/radiance_upscale.js`  
**Files added (workflows):** `workflows/LTX 2.3 (Test Workflow).rad`, `workflows/LTX 2.3 (Two 2x Latent Upscales Test Workflow).rad`

---

## Detailed Changes

---

### `image/upscale.py`

#### SUPIR diffusion upscaler integration *(changes relative to the original file without SUPIR)*

SUPIR (v0F / v0Q) is a latent-diffusion upscaler — Spandrel raises `UnsupportedModelError`
for it. A dedicated code path delegates inference to the ComfyUI-SUPIR extension (kijai/ComfyUI-SUPIR).

**New globals:**
- `_SUPIR_MODELS = {"SUPIR-v0F_fp16", "SUPIR-v0Q_fp16"}`
- `_find_supir_dir()`: locates the ComfyUI-SUPIR extension directory

**`AI_MODELS` list:** `"SUPIR-v0F_fp16"` and `"SUPIR-v0Q_fp16"` added

**`INPUT_TYPES` (optional section):**
- Added `supir_steps` (INT, default 45, range 1–200) — number of diffusion steps
- Added `sdxl_model_name` (STRING) — SDXL base checkpoint filename; auto-detected if empty
- Added `supir_prompt` (STRING, multiline) — text conditioning for SUPIR

**`_load_model()`:**
- Added `sdxl_model_name` parameter
- Routes `_SUPIR_MODELS` filenames to `_load_supir_model()` instead of Spandrel
- Improved Spandrel error: replaces silent empty `UnsupportedModelError` (str(e) == "") with a human-readable diagnostic

**`_load_supir_model()` *(new method)*:**
- Looks up `SUPIR_model_loader` in `nodes.NODE_CLASS_MAPPINGS` (correct key — old approach used wrong key name)
- Fallback: searches ComfyUI-SUPIR module by directory path for ComfyUI 0.19.x
- Auto-detects SDXL base checkpoint from `checkpoints/` folder if `sdxl_model_name` is empty
- Temporarily registers the SUPIR model directory in `folder_paths` so the loader can find it
- Caches result as `("supir", SUPIRMODEL, SUPIRVAE, cls_map)` tuple

**`_run_supir()` *(new method)*:**
- Runs the four ComfyUI-SUPIR stages in sequence: encode → condition → sample → decode
- Uses `inspect.signature` to forward only the kwargs each node method actually accepts
- Accepts `steps` parameter forwarded from the UI widget
- Sampling defaults: `cfg_scale=4.0`, `EDM_s_churn=5`, `s_noise=1.003`, `sampler="RestoreEDMSampler"`

**`_fallback_upscale()`:**
- Changed from whole-batch `F.interpolate` to per-frame loop — avoids misleading memory warning that included the batch dimension in the estimate

**`upscale()`:**
- Added `supir_steps`, `sdxl_model_name`, `supir_prompt` parameters
- SUPIR models routed to `_run_supir()` and bypass the HDR compress/expand pipeline (input passed as-is)

---

### `js/radiance_upscale.js` *(new file)*

Frontend companion for `RadianceAIUpscale`. Manages dynamic widget visibility.

- Hides `supir_steps`, `sdxl_model_name`, `supir_prompt` when a non-SUPIR model is selected
- Three-mechanism visibility (same pattern as `radiance_io.js`):
  `widget.options.hidden` + `widget.hidden` + `type="hidden"` + `computeSize=[0,-4]`
  + `draw=()=>{}` + DOM `display:none` + `computedHeight=4`
- `refreshNodeSize()`: uses `node.size = [w, h]` (new array) so Vue 3 reactive proxy detects the change
- Two-pass + polling resize strategy: 100ms initial hide, 500ms size correction, 5× polling retry at 200ms intervals for deferred Vue layouts

#### Widget visibility restore on page refresh / tab switch (Nodes 2.0 fix)
- Added `onConfigure` hook with `setTimeout(fn, 10)` — fires after `widgets_values` are restored, schedules re-evaluation before Vue settles reactivity
- Added `loadedGraphNode` hook with `setTimeout(fn, 100)` — fallback for the full saved-workflow restore path
- Added `async afterConfigureGraph()` hook with direct call (no inner `setTimeout`) — at this point Vue has crossed its `await` boundary and `node.widgets` is a reactive proxy; direct `splice(0,0)` triggers correct re-render
- `_radianceUpdateVisibility` stored on node instance for all three hooks to share

---

### `nodes_io.py` + `nodes_qc.py`

#### Path inputs: surrounding quotes stripped automatically
- Windows "Copy as path" (Shift+Right-click) wraps paths in double-quotes — these are now accepted as-is
- `_strip_path_quotes()` helper added to both files; strips leading/trailing `"` and `'` plus whitespace
- Applied to: `RadianceDigitalCinemaRead.source_path`, `RadianceWrite.output_path` / `remote_path`, `RadianceEXRWriteMultipart.output_path` / `remote_path`, `QCReportExporter.output_path`
- Forward slashes and backslashes continue to work unchanged
- Tooltips updated on all affected path fields

### `nodes_io.py`

#### `bit_depth` widget: restricted to EXR only
- Previously visible for EXR and PNG — PNG encodes its depth in the format name ("PNG (8-bit)" / "PNG (16-bit)"), `bit_depth` was silently ignored for PNG and Radiance HDR
- Now hidden for PNG and HDR formats; only shown for EXR (Half Float vs Full Float) — fix in `radiance_io.js`

#### PNG alpha channel support
- `alpha_mode` now fully functional for PNG (was EXR-only)
- 8-bit PNG: written via PIL as RGBA + tEXt metadata chunks
- 16-bit PNG: written via cv2 as BGRA uint16 (alpha supported; tEXt metadata not written — cv2 limitation)
- Alpha extraction refactored: moved before the format if/elif chain, shared by EXR and PNG paths

#### PNG tEXt metadata (`custom_metadata`)
- 8-bit PNG: metadata written as tEXt chunks via PIL `PngInfo`
- 16-bit PNG: metadata not written (cv2 limitation — use EXR for full metadata support)
- PIL import added with `_HAS_PIL` flag + `_build_pnginfo()` helper

#### `broadcast_safe` default changed to `False`
- INPUT_TYPES default: `True` → `False`
- Python function signature: `broadcast_safe=True` → `False` (aligned with UI default for consistency)
- Python gating on `is_display_space` unchanged — `broadcast_safe` has no effect outside sRGB anyway

#### `RadianceDigitalCinemaRead` — `frame_number` video seeking implemented
- Previously declared in INPUT_TYPES and shown in Single Frame mode but never read in Python
- New code path at the top of the Single Frame section: detects video via `_is_video_file()`, opens via `VideoCapture`, seeks with `cap.set(CAP_PROP_POS_FRAMES, frame_number - 1)`, clamps to total frame count
- Image sources (EXR, PNG, etc.): `frame_number` is silently ignored — a still image has exactly one frame
- FPS returned: `detected_fps` from the video container (not `fps_override`, which is disabled in Single Frame mode)

#### `RadianceDigitalCinemaRead` — `fps_override` disabled in Single Frame mode
- Python: `fps = fps_override if fps_override > 0 else 24.0` → `fps = 24.0` (hardcoded)
- JS: already hidden via `!isSingle` — no change needed
- FPS has no semantic meaning for a single still image

---

#### Audio export: new `write_external_audio_file` dropdown
- Replaces the old boolean `extract_audio_wav`
- Options: `"None"`, `"WAV — PCM 32-bit Float"`, `"WAV — PCM 24-bit"`, `"WAV — PCM 16-bit"`, `"AIFF — PCM 24-bit"`, `"FLAC — Lossless"`
- WAV 32-bit Float: pure Python implementation (no ffmpeg dependency)
- All other formats: ffmpeg subprocess (raw f32le pipe → target codec)

#### New `audio_filename_suffix` widget
- Default `"_audio"` — appended to `filename_prefix` for the exported audio file

#### Audio export directory fix
- `_write_audio_file()` is now called **after** image writes so the sequence subfolder (`prefix_timestamp/`) is resolved first

#### New `_write_audio_file()` method
- Handles all lossless audio formats (WAV 32-bit float natively, others via ffmpeg)

#### New `_write_animated()` method
- Handles animated GIF and WEBP in Sequence mode, static GIF/WEBP in Single Image mode
- Uses `imageio.v3`, duration derived from fps

#### Format routing rework (GIF/WEBP)
- GIF/WEBP moved out of Video group:
  - `"GIF — Animated"` / `"WEBP — Animated"` → Sequence mode only
  - `"GIF"` / `"WEBP"` (static) → Single Image mode only
  - Video mode is now strictly for cinema codecs (H.264, H.265, ProRes variants)

#### H.264 encoder fix
- Switched from `imageio` (no quality control) to `ffmpeg` subprocess with `-crf`
- `quality 0–100` maps to CRF `51–0` (mirroring existing H.265 path)

#### JPEG quality scale fix
- Removed erroneous `×10` multiplier — `cv2.IMWRITE_JPEG_QUALITY` expects `0–100` directly

#### Quality default raised
- Default: `10` → `80`

#### EXR loader error handling
- Split bare `except: pass` into `except ImportError: pass` + `except Exception as e: logger.warning(...)`

#### Detailed widget tooltips added
- `quality`, `output_color_space`, `broadcast_safe`, `output_path`, `compression`, `alpha_mode`, `custom_metadata`

#### EXR multipart import fix
- Moved `import OpenEXR` inside the `write_exr_multipart()` call scope
- Added `OPENCV_IO_ENABLE_OPENEXR=1` environment variable guard

#### `RadianceDigitalCinemaRead` — Single Frame mode fix
- `is_video` corrected from `True` to `False` — was incorrectly routing image files through the video decoder (cv2.VideoCapture)
- Added a dedicated Single Frame path for image files (EXR, HDR, PNG, JPG) with correct bit-depth normalisation (uint8/uint16/float32)

#### `input_colorspace` default changed
- `"sRGB (Standard)"` → `"Linear (sRGB)"` — better default for Radiance's HDR/EXR-first workflow

---

### `js/radiance_io.js`

#### `FORMAT_GROUPS` constant
```js
{ "Video": [...], "Sequence": [...], "Single Image": [...] }
```
Mirrors `WRITE_FORMATS` from `nodes_io.py`. GIF/WEBP placed correctly per mode.

#### `setWidgetVisible()` function (full implementation)
Four-mechanism visibility supporting both LiteGraph canvas and Nodes 2.0:
1. `widget.options.hidden = !visible` — Nodes 2.0 Vue filter
2. `type="hidden"` + `computeSize=[0,-4]` — collapses height for all widget types
3. `widget.draw = () => {}` — prevents text bleeding on STRING widgets
4. `inputEl/element display:none` — hides DOM node for customtext widgets
- `widget.computedHeight = 4` on hide; `_origComputedHeight` saved and restored on show

#### Dynamic format filtering
- `formatWidget.options.values` updated on every `write_mode` change
- If current format is no longer valid for the new mode, resets to first valid option

#### Dynamic quality label
- `"quality (CRF)"` — H.264 / H.265
- `"quality (0–100)"` — JPEG / WEBP
- `"quality"` — all others

#### Quality widget visibility
- Hidden for formats that ignore it: EXR, PNG, ProRes, Radiance HDR

#### Audio suffix visibility
- `audio_filename_suffix` hidden when `write_external_audio_file = "None"`

#### Vue node sizing fix
- `refreshNodeSize()`: changed direct array mutation to new array assignment (`node.size = [newW, newH]`)
- Direct mutation bypasses Vue 3's proxy setter — caused blank gaps after widget hide/show

#### `refreshNodeSizeWithRetry()` *(new function)*
- Calls `refreshNodeSize()` immediately, then polls up to 5× at 200ms intervals
- Applied to both the Read node and Write node

#### Read node widget visibility *(new section)*
- Added `RadianceDigitalCinemaRead` handler — hides `start_frame`, `frame_limit`, `fps_override` in Single Frame mode; shows `frame_number` instead

#### `bit_depth` visibility fix
- `setWidgetVisible(bitDepthWidget, isSeqLike && (is_exr || is_png), node)` → `isSeqLike && is_exr`
- PNG encodes depth in the format name; Radiance HDR is always 32-bit RGBE — neither uses `bit_depth`

#### Write node: `frame_padding` visibility
- `frame_padding` hidden for `"Video"` and `"Single Image"` modes (only relevant for `"Sequence"`)

#### Widget visibility restore on page refresh / tab switch (Nodes 2.0 fix)
- Same three-hook pattern as `radiance_upscale.js` — `onConfigure` + `loadedGraphNode` + `afterConfigureGraph`
- `afterConfigureGraph` also covers `radiance_sampler.js` and `radiance_upscale.js` nodes via the shared `_radianceUpdateVisibility` scan

---

### `nodes_loader.py`

#### LTX model detection heuristic fix (v3.1)
- **Bug**: `patch_embedding.weight` was used as the primary LTX key at position 4, BEFORE the Wan heuristic at position 6. Since Wan 2.1 also has `patch_embedding.weight`, ALL Wan models were silently misdetected as LTX
- **Fix**: Moved Wan heuristic before LTX; changed LTX primary key to `patchify_proj` (unique to LTXV architecture, not present in Wan/Flux/SD3/PixArt)
- Added LTX fallback: older LTXV checkpoints detected via `patch_embedding` + `adaln_single` + NO `time_embedding`
- Fixed PixArt detection: added exclusion of `patchify_proj` (previously matched LTX by mistake)
- `_SAFETENSORS_PEEK`: `80` → `200` (read more keys for reliable heuristics)

#### New `_FORMAT_MAP` (architecture-prefixed labels)
- Previous code returned bare channel counts like `"16ch"` — didn't match format labels in `vae.py`'s `LATENT_FORMAT_MAP`
- New labels: `"flux_16ch"`, `"ltxv"`, `"ltxav"`, `"hunyuan_16ch"`, `"wan_16ch"`, `"sd_4ch"`, etc.

#### New download presets (7 entries for LTX 2.3)
- `ltx-2.3-22b-dev.safetensors` — main diffusion model (bf16)
- `ltx-2.3-22b-dev-fp8.safetensors` — fp8 variant
- `gemma_3_12B_it.safetensors` — Gemma 3 12B text encoder
- `gemma_3_12B_it_fp4_mixed.safetensors` — Gemma 3 fp4 mixed variant
- `ltx-2.3_text_projection_bf16.safetensors` — text projection model (Kijai)
- `LTX23_video_vae_bf16.safetensors` — video VAE (Kijai)
- `LTX23_audio_vae_bf16.safetensors` — audio VAE (Kijai)

#### New `ltxav` model type (LTX-Video 2.3 audio-video)
- Added `"ltxav"` to `_LATENT_CHANNELS`: value `128` (same as `ltxv`)
- Renamed model type `"ltx"` → `"ltxv"`; updated `_FORMAT_MAP` labels
- Added `"ltxav"` to the CLIP_SLOTS map with `["llm_encoder", "text_projection"]`
- Reassigned `"ltxv"` CLIP_SLOTS from `["t5xxl"]` → `["llm_encoder", "text_projection"]`

#### `text_projection` CLIP slot support
- Added `text_projection` parameter to `_assemble_clip_paths()` and all callers
- Added `text_projection` widget in `INPUT_TYPES`

#### Preset ordering fix + new LTX 2.3 presets
- Added `"→ LTX Video 2.3"` and `"→ LTX Video 2.3 (Low VRAM)"` presets
- Presets now only override `model_type` — no longer force `weight_dtype`, `clip_dtype`, or `offload_mode`

#### New outputs: `AUDIO_VAE` and `LATENT_UPSCALE_MODEL`

#### New `audio_vae_name` widget
- Dropdown: checkpoints + vae folders + `"Baked Audio VAE (from UNET)"`
- Baked extraction: loads `AudioVAE` from the UNET state dict

#### New `upscale_model_name` widget
- Dropdown from `latent_upscale_models` folder
- Auto-detects architecture: HunyuanVideo 720p (`blocks.0.block.0.conv.weight`), HunyuanVideo 1080p (`up.0.block.0.conv1.conv.weight`), LTX-Video latent upsampler (`post_upsample_res_blocks`)

#### VAE loading: metadata-aware for LTX 2.3
- Uses `comfy.utils.load_torch_file(path, return_metadata=True)` and `comfy.sd.VAE(sd=sd, metadata=metadata)`
- Required for LTX 2.3's 256-channel VAE (without metadata ComfyUI falls back to 128-ch config)

#### Native Gemma 3 DualCLIP loading
- Constructs DualCLIP path using `CLIPType.LTXV` for Gemma + `ltxv`/`ltxav` model types
- Combines `llm_encoder` + `text_projection` standalone file, or extracts projection natively from UNET

---

### `js/radiance_loader.js` *(new file)*

Frontend companion for the updated loader node — dynamic widget updates for `audio_vae_name`, `upscale_model_name`, and `text_projection`.

---

### `nodes_resolution.py`

#### New resolution presets
- `"LTX Portrait (768×1024)"`, `"LTX 720p (1280×736)"`, `"LTX 720p (4x latent upscale wf) (1280×768)"`
- `"LTX 1080p (1920×1088)"`, `"LTX 1080p (4x latent upscale wf) (1920×1152)"`
- `"LTX 2K DCI (2048×1088)"`, `"LTX 2K DCI (4x latent upscale wf) (2048×1152)"`
- `"LTX 4K UHD (3840×2176)"`, `"LTX 4K DCI (4096×2176)"`
- Removed old `"LTX 720p (1216×704)"` and `"LTX Portrait (704×1216)"`

#### New widget: `crop_to_broadcast_resolution` — crops padded heights for broadcast compliance
#### New widget: `frame_computation` — `"Manual (Frames)"` or `"Auto (Seconds)"`
#### New widget: `duration_seconds` (float, default 5.0)
#### New output: `crop_bbox` (`BOUNDING_BOX`) — connects directly to `RadianceVAE4KDecode`

---

### `js/radiance_resolution.js`

- Forces `latent_channels = 128` for LTX presets; adds `"LTXV"` to model type options
- Full Nodes 2.0 widget visibility compatibility (`options.hidden`, `computedHeight`, deferred init)

---

### `nodes_sampler.py`

#### Changes v2.3.2 → v2.3.3 *(features absent from v2.2 reference file, confirmed user additions)*

##### Full NestedTensor infrastructure
- `from comfy.nested_tensor import NestedTensor as _NestedTensor` + `_HAS_NESTED_TENSOR` flag (try/except ImportError guard)
- `log_tensor()` helper updated to detect and log NestedTensor sub-shapes
- `isinstance(current_latent, _NestedTensor)` check in sampling loop — triggers NestedTensor-aware noise reconstruction
- `stage_noise = _NestedTensor(tuple(new_noises))` — per-component noise rebuild for LTX-AV NestedTensor latents
- Two guards against `AttributeError` on `.is_cuda` (tagged `# ALBABIT-FIX`) — NestedTensor lacks this attribute

##### New `ltxav` model type (full support)
- Added `"ltxav"` to `MODEL_TYPES`, `VIDEO_MODEL_TYPES`, `CFG_GUIDED_MODELS`
- Added `"ltxav"` entry in `MODEL_DEFAULTS`: `cfg=3.0, scheduler="beta", shift=3.0, guidance_type="cfg"`
- `detect_model_type()`: detection via `recombine_audio_and_video_latents` attribute or `"ltxav"` in class name; `"LTXAV"` → `"ltxav"` in format map
- `is_ltx_av` flag and `ltxav_obj` reference established at sampling start
- Scheduler compatibility warnings for LTX 2.3: Karras, Exponential, `uni_pc`, and non-Gaussian noise types

##### LTX-AV audio latent recombination
- `separate_audio_and_video_latents()` → sample + scale video/audio components → `recombine_audio_and_video_latents()` called on final output
- Gated by `is_ltx_av and ltxav_obj is not None and _HAS_NESTED_TENSOR`

##### New presets: `"▶ LTX 2.3 LowRes (32 steps)"` and `"▶ LTX 2.3 HighRes (40 steps)"`
- Added to `LTX_PRESETS` list and `PRESET_DEFAULTS` dict
- Both use `model_type="ltxav"`, `scheduler="beta"`, `force_full_denoise_steps=True`, `force_exact_steps=True`

##### New `force_full_denoise_steps` and `force_exact_steps` parameters
- `force_full_denoise_steps` (bool): forces final sigma to `0.0` even on truncated i2i schedules
- `force_exact_steps` (bool): back-calculates full schedule length so truncated slice has exactly `steps` entries
- Both added to `INPUT_TYPES` and `compute_base_sigmas()` signature

##### `import gc` + pre-flight cleanup (tagged `# ALBABIT FIX`)
- GC sweep before sampling to reclaim fragmented VRAM

##### `sigmas_override` improvements (tagged `# ALBABIT-FIX`)
- Console alert printed when `sigmas_override` is connected, warning that UI step/denoise/scheduler widgets are ignored
- `total_steps` synced to `len(sigmas_override) - 1` to prevent `SigmaIndexer` out-of-bounds
- `_get_base_sigmas()` returns overridden sigmas directly (bypasses schedule generation) to guarantee indexer size match

#### Changes v2.3.3 → current *(diff "Radiance Originals Files" → current working tree)*

##### `force_full_denoise_steps` → `terminal_sigma_to_zero`
- Renamed widget and parameter throughout
- Behavior clarified: sigma schedule is **always** truncated based on `denoise` — `terminal_sigma_to_zero` only forces the final sigma to `0.0`, it no longer bypasses truncation

##### LTX 2.3 LowRes default steps: 32 → 20
- `PRESET_DEFAULTS` entry for LTX 2.3 LowRes: `steps: 32` → `steps: 20`

##### LTX-AV detection log message updated
- `"Volume adjustment will be applied at the end of the phase"` → `"Latents will be processed natively"`

##### Removed `video_normalization_factors` and `audio_normalization_factors`
- Both UI widgets and their parsing/application code removed — normalization is now handled natively by `recombine_audio_and_video_latents()`

---

### `js/radiance_sampler.js`

#### Changes v2.3.2 → v2.3.3 *(ALBABIT-tagged lines in "Radiance Originals Files")*

- Replaced the floating `div` sigma-connection indicator with a native embedded text widget (Nodes 2.0 compatible)

#### Changes v2.3.3 → current *(diff "Radiance Originals Files" → current working tree)*

- Preset renamed: `"▶ LTX 2.3 LowRes (32 steps)"` → `"▶ LTX 2.3 LowRes (20 steps)"`; `steps` field updated `32 → 20`
- Preset objects: `force_full_denoise_steps` key replaced by `terminal_sigma_to_zero` (matches renamed Python widget)
- Added full `setWidgetVisible()` implementation with Nodes 2.0 support (`widget.options.hidden`, `computedHeight`, DOM `display:none`) — same four-mechanism pattern as `radiance_io.js`
- Widget visibility wired for `cond_weight_b` (hidden unless `multi_cond_mode` is active) and tile widgets (`tile_size`, `tile_overlap`, `tile_blend`)
- Added `terminal_sigma_to_zero` to the list of widgets auto-hidden when a sigma override is connected
- `checkSigmaConnection()` method added — monitors sigma input connection via `onDrawBackground` and toggles widget visibility in real time

#### Widget visibility restore on page refresh / tab switch (Nodes 2.0 fix)
- Same three-hook pattern as `radiance_upscale.js` — `onConfigure` + `loadedGraphNode` + `afterConfigureGraph`
- `samplerRestoreFn` stored as `_radianceUpdateVisibility`; calls `updateUILocks`, `updateDescription`, `toggleDynamicFields`, `checkSigmaConnection`

---

### `nodes_depth.py`

#### VRAM / OOM fix for Depth Anything V2
- Per-frame GPU tensor release: `del outputs`, `del inputs`, `.cpu()` calls inside depth loop

---

### `hdr/vae.py`

#### New inputs: `temporal_size` and `temporal_overlap`
- Chunk-based 3D VAE temporal tiling with cosine blending across overlap zones
- Prevents VRAM OOM and flickering on long videos with LTX 2.3's 3D VAE

#### New input: `crop_bbox` (`BOUNDING_BOX`, `forceInput`)
- Crops decoded image to broadcast resolution directly inside the decode node

---

### `workflows/`

- `LTX 2.3 (Test Workflow).rad` — Single-pass LTX 2.3 generation workflow
- `LTX 2.3 (Two 2x Latent Upscales Test Workflow).rad` — LTX 2.3 with two successive 2× latent upscale passes

---

## Commit History

| Hash | Description |
|------|-------------|
| `9131d2f` | fix: widget visibility restore + path quote stripping (JS Nodes 2.0 + Python) |
| `951ecf2` | fix(nodes_io): PNG alpha+metadata, video frame seeking, bit_depth EXR-only |
| `a3493ea` | feat(upscale): SUPIR integration + radiance_upscale.js + Read node Single Frame fix + io sizing |
| `374a138` | fix(nodes_io): separate write_exr_multipart import + add OPENCV_IO_ENABLE_OPENEXR |
| `e0d52af` | fix(nodes_io): merge developer bug fixes and memory management improvements |
| `d5033cf` | fix(js): Nodes 2.0 widget visibility across all three UI extensions |
| `9f02852` | refactor: unify LTX identifiers (ltx→ltxv), clean format labels, rename workflows |
| `5af19ff` | fix: VRAM OOM fix depth node, EXR error handling, quality default, crop_bbox in VAE decode |
| `955e2d6` | fix: correct LTX-Video 128ch latent format across loader, resolution, sampler |
| `fa2ac44` | fix(radiance-write): fix audio export dir, output_path tooltip, quality tooltip |
| `ef4aa44` | fix(radiance-write): raise quality default to 80, restore dynamic label |
| `df90b64` | fix(radiance-write): wire quality to H.264 CRF and fix JPEG quality scale |
| `ee6c833` | feat(radiance-write): add tooltips to INPUT_TYPES widgets |
| `ae92672` | feat: LTX-Video 2.3 support, bug fixes and UI improvements |

---

## Compatibility Notes

- **No breaking changes** to node names or mandatory inputs for existing Flux/WAN/HunyuanVideo workflows.
- `write_external_audio_file` dropdown **replaces** the old `extract_audio_wav` boolean — existing saved workflows using the boolean will need to be updated manually.
- `alpha_mode` is now functional for **EXR** (RGBA float) and **PNG** (8-bit RGBA via PIL, 16-bit RGBA via cv2). Not implemented for TIFF, JPEG, Radiance HDR, or video formats.
- The `"Baked VAE (from UNET)"` default for the VAE widget is a behavior change: users who previously connected an explicit VAE file will need to re-select it.
- SUPIR requires the ComfyUI-SUPIR extension (kijai/ComfyUI-SUPIR) to be installed via ComfyUI Manager.
