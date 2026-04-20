# Radiance — Changes by Albabit
**Branch:** `ltxv23-support-and-various-fixes`  
**Fork:** https://github.com/Albabit/radiance/tree/ltxv23-support-and-various-fixes  
**Based on:** `fxtdstudios/radiance` v2.3.3  
**Author:** [@Albabit](https://github.com/Albabit)  
**Date:** April 2026

---

## Overview

This branch adds full LTX-Video 2.3 support (model loading, audio-video latent handling, correct 128-channel format), fixes several bugs in the Write node (quality control, audio export, H.264 encoder), improves the Resolution node (new presets, crop output, smart latent calculation), and adds temporal decoding to the VAE decode node.

**Files modified:** `nodes_loader.py`, `nodes_resolution.py`, `nodes_sampler.py`, `nodes_io.py`, `nodes_depth.py`, `hdr/vae.py`  
**Files modified (JS):** `js/radiance_io.js`, `js/radiance_resolution.js`, `js/radiance_sampler.js`  
**Files added (JS):** `js/radiance_loader.js`  
**Files added (workflows):** `workflows/LTX 2.3 (Test Workflow).rad`, `workflows/LTX 2.3 (Two 2x Latent Upscales Test Workflow).rad`

---

## Detailed Changes

---

### `nodes_loader.py`

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
- Renamed model type `"ltx"` → `"ltxv"`; updated `_FORMAT_MAP` labels to `"ltxv"` / `"ltxav"` (simplified from intermediate `"ltx_128ch"` / `"ltxav_128ch"`)
- Added `"ltxav"` to the CLIP_SLOTS map with `["llm_encoder", "text_projection"]`
- Reassigned `"ltxv"` CLIP_SLOTS from `["t5xxl"]` → `["llm_encoder", "text_projection"]`
- Added `"text_projection"` slot type in the CLIP_SLOTS comment header
- Added `"ltxav"` to the valid model types list and to the arch-detection loop

#### `text_projection` CLIP slot support
- Added `text_projection` parameter to `_assemble_clip_paths()`
- Updated all callers of `_assemble_clip_paths()` to pass `text_projection`
- Added `text_projection` widget (`clip_slot`) in `INPUT_TYPES` with tooltip: *"Text Projection Model. Used with Gemma 3 for native DualCLIP loading in LTX 2.3."*
- Updated `llm_encoder` tooltip to mention Gemma 3 and LTX 2.3

#### Preset ordering fix + new LTX 2.3 presets
- Moved `"→ LTX Video 13B"` after the LTX 2.3 entries (was incorrectly placed before)
- Added `"→ LTX Video 2.3"` preset: `model_type=ltxav`, `weight_dtype=bf16`, `clip_slots={"llm_encoder": True}`
- Fixed `"→ LTX Video 2.3 (Low VRAM)"`: `model_type` ltxv→ltxav, `clip_slots` t5xxl→llm_encoder
- Added VRAM estimates for `ltxav`: `20.0 GB` (full) and `8.0 GB` (offload)

#### Preset behavior change
- Presets **only override `model_type`** now — they no longer force `weight_dtype`, `clip_dtype`, or `offload_mode`, respecting the user's UI choices

#### New outputs: `AUDIO_VAE` and `LATENT_UPSCALE_MODEL`
- `RETURN_TYPES` extended: `("MODEL", "CLIP", "VAE", "VAE", "CONTROL_NET", "LORA_STACK", "LATENT_UPSCALE_MODEL", ...)`
- `RETURN_NAMES` extended: `(..., "AUDIO_VAE", ..., "upscale_model", ...)`

#### New `audio_vae_name` widget
- Dropdown populated from both `checkpoints` and `vae` folders, plus `"Baked Audio VAE (from UNET)"` option
- **Baked extraction:** if selected, loads `AudioVAE` from the UNET file using `comfy.utils.load_torch_file(..., return_metadata=True)` and instantiates `comfy.ldm.lightricks.vae.audio_vae.AudioVAE`
- **Standalone loading:** if a file is selected, loads it independently from `checkpoints/` or `vae/`
- Cache-aware (mtime + size fingerprint)

#### New `upscale_model_name` widget
- Dropdown populated from `latent_upscale_models` folder
- Supports auto-detection of model architecture via state dict keys:
  - HunyuanVideo 720p (`HunyuanVideo15SRModel`)
  - HunyuanVideo 1080p (variant with `up.*.block.*.conv1`)
  - LTX-Video latent upsampler (`LatentUpsampler`, loaded with config from metadata)
- Cache-aware

#### VAE loading: metadata-aware for LTX 2.3
- Standalone VAE now uses `comfy.utils.load_torch_file(path, return_metadata=True)` and `comfy.sd.VAE(sd=sd, metadata=metadata)` — required for LTX 2.3's 256-channel VAE architecture (without metadata ComfyUI falls back to 128-channel LTX 1.0 config and raises a size mismatch)
- Default VAE option changed to `"Baked VAE (from UNET)"` with extraction logic via `load_checkpoint_guess_config`
- Re-evaluates UNET cache hit if baked VAEs fell out of cache

#### Native Gemma 3 DualCLIP loading
- When `llm_encoder` contains a Gemma model with `ltxv`/`ltxav` model type, constructs a DualCLIP path using `CLIPType.LTXV`
- Combines `llm_encoder` + `text_projection` (standalone file) or falls back to extracting the projection natively from the UNET file

#### Bug fixes
- Safe `or []` on `folder_paths.get_filename_list()` calls in both `RadianceLoader` and the internal loader — prevents Python type crash when folders are empty on first run

---

### `js/radiance_loader.js` *(new file)*

Frontend companion for the updated loader node. Handles dynamic widget updates for the new `audio_vae_name`, `upscale_model_name`, and `text_projection` inputs.

---

### `nodes_resolution.py`

#### New resolution presets
- `"LTX Portrait (768×1024)"` — 3:4
- `"LTX 720p (1280×736)"` — 16:9 padded
- `"LTX 720p (4x latent upscale wf) (1280×768)"` — for 4× latent upscale workflows
- `"LTX 1080p (1920×1088)"` — padded (replaces old 1080p)
- `"LTX 1080p (4x latent upscale wf) (1920×1152)"` — for 4× latent upscale workflows
- `"LTX 2K DCI (2048×1088)"` — 1.90:1 padded
- `"LTX 2K DCI (4x latent upscale wf) (2048×1152)"` — padded (1152 is multiple of 128)
- `"LTX 4K UHD (3840×2176)"` — 16:9 padded
- `"LTX 4K DCI (4096×2176)"` — 1.90:1 padded
- Removed the old `"LTX 720p (1216×704)"` and `"LTX Portrait (704×1216)"` entries
- Renamed `"LTX 1080p (1920×1088)"` as padded to clarify broadcast compliance requires crop

#### New `LTXV` model type
- Added `"LTXV"` to `MODEL_TYPES` and `LATENT_CHANNELS` (→ 128)
- Renamed `"Auto (Flux 16ch)"` → `"Auto (Flux 16ch / LTXV)"` (auto-detects LTX presets)
- Auto-detection: if model_type is "Auto" and preset name contains "ltx", forces `latent_c = 128`

#### New widget: `crop_to_broadcast_resolution`
- Boolean, default `True`
- Computes a bounding box to crop padded heights (1088→1080, 768→720, 1152→1080, 2176→2160) for broadcast compliance

#### New widget: `frame_computation`
- Dropdown: `["Manual (Frames)", "Auto (Seconds)"]`
- When `"Auto (Seconds)"`, calculates frame count from `duration_seconds × fps`, aligned to the model's temporal stride:
  - LTX: stride=8 → `(n×8)+1`
  - WAN / Hunyuan: stride=4 → `(n×4)+1`
  - Others: stride=4 (safe fallback)

#### New widget: `duration_seconds`
- Float input (default `5.0`, min `0.1`, max `120.0`, step `0.1`)

#### New output: `crop_bbox` (`BOUNDING_BOX`)
- Returns `{"x": x, "y": y, "width": w, "height": h}` — ready to connect to `RadianceVAE4KDecode.crop_bbox` or `ImageCropV2`

#### `video_frames` default changed
- Default: `81` → `121`

#### `scale_factor` precision change
- `min`: `0.25` → `0.1`
- `step`: `0.25` → `0.01` (allows exact values like 0.25 from JS without rounding errors)

#### Smart latent compression (spatial + temporal)
- LTX-Video: **32× spatial**, **8× temporal** compression
- WAN / HunyuanVideo: **8× spatial**, **4× temporal** compression
- Image models (Flux, SDXL, SD 1.5): **8× spatial**, no temporal

#### Correct temporal latent frame count
- Formula: `lat_t = (frames - 1) // temporal_scale + 1` (standard 3D VAE equation)
- Previous code used `actual_batch` directly as the T dimension, which was incorrect for video models

#### Fix: latent format derivation for manual channel override
- Previous: `latent_c >= 16` → `"flux"` (wrong for LTXV 128ch)
- New: `latent_c >= 128` → `"ltxv"`, `latent_c >= 16` → `"flux"`, else `"sdxl"`

---

### `js/radiance_resolution.js`

- Mirror of Python: force `latent_channels = 128` for LTX presets in the UI
- Adds `"LTXV"` to the model type dropdown options

#### Nodes 2.0 widget visibility fixes
- `setWidgetVisible()` now supports both legacy LiteGraph canvas AND ComfyUI Nodes 2.0 (Vue-based rendering):
  - `widget.options.hidden = !visible` — primary Nodes 2.0 mechanism (Vue filter: `t.filter(e => !(e.options?.hidden || ...))`)
  - `widget.hidden = !visible` — legacy LiteGraph `getLayoutWidgets()` exclusion
  - `widget.computedHeight = 4` on hide — collapses Vue CSS height to 0px (`height = computedHeight - 4`)
  - `widget.computedHeight = _origComputedHeight` on show (fallback: `32`) — restores correct Vue row height
  - `node.widgets.splice(0, 0)` unconditionally — triggers Vue reactive proxy to re-evaluate `options.hidden`
- `isVideo` check extended to include integer `1` (Nodes 2.0 stores toggle values as `0`/`1`, not `true`/`false`)
- Initial `toggleFields()` call **deferred to `setTimeout(100ms)`** (was synchronous at node creation) — ensures Vue completes its first layout pass and sets `computedHeight` on all widgets before any are hidden; restoring a widget hidden before Vue's first render caused 0px blank-space in Nodes 2.0

---

### `nodes_sampler.py`

#### New `ltxav` model type support
- Added `"ltxav"` to `MODEL_TYPES`, `VIDEO_MODEL_TYPES`, and `CFG_GUIDED_MODELS`
- Added `"ltxav"` to `MODEL_DEFAULTS`:
  ```python
  "ltxav": { "cfg": 3.0, "scheduler": "beta", "guidance": 0.0,
              "shift": 3.0, "sampler": "euler", "guidance_type": "cfg" }
  ```
- Added `ltxav` detection in `detect_model_type()`:
  - Checks for `"ltxav"` in class name or `recombine_audio_and_video_latents` attribute
  - Added `"LTXAV"` to the config class name map

#### LTX-AV audio latent recombination
- Added logic to extract the `ltxav_obj` (inner diffusion model) during sampling
- Uses `ltxav_obj.separate_audio_and_video_latents()` and `recombine_audio_and_video_latents()` on the final output
- **Removed normalization multipliers** — LTX 2.3 handles recombination cleanly without manual scaling

#### Extended channel validation for LTXV
- Previous code checked `"ltx*"` format strings and expected 16ch, triggering false warnings
- Now checks `"ltxav"`, `"ltxv"` keywords first before channel count validation

#### LTX 2.3 sampler presets in `SAMPLER_PRESETS`
- `"→ LTX Video 2.3"`: `model_type=ltxav`, `scheduler=beta`, `cfg=3.0`, `shift=3.0`
- `"→ LTX Video 2.3 (Low VRAM)"`: same but with memory offload

#### Bug fix: sigma schedule truncation
- Fixed: `force_full_denoise_steps` was incorrectly bypassing truncation
- Now: schedule is ALWAYS truncated based on `denoise`; `force_full_denoise_steps` only forces the final sigma to 0.0

#### Step count display
- Changed progress/log display from 0-based to 1-based step count (1 to N) for clarity

#### Pre-flight GPU memory cleanup
- Added `import gc` and `gc.collect()` before sampling start

---

### `js/radiance_sampler.js`

#### New LTX 2.3 presets
- `"▶ LTX 2.3 LowRes (20 steps)"`: `steps=20`, `cfg=3.0`, `scheduler=beta`, `denoise=1.0`, `shift=3.0`, `model_type=ltxav`, `terminal_sigma_to_zero=true`, `force_exact_steps=true`
- `"▶ LTX 2.3 HighRes (40 steps)"`: `steps=40`, `cfg=3.0`, `scheduler=beta`, `denoise=0.45`, `shift=6.0`, `model_type=ltxav`
- Note: previous version had `"▶ LTX 2.3 LowRes (32 steps)"` — renamed and revised to 20 steps with updated fields

#### `setWidgetVisible()` function — Nodes 2.0 compatibility
- Physically hides/shows widgets (type="hidden" + computeSize=[0,-4]) for LiteGraph canvas
- `widget.options.hidden = !visible` — Nodes 2.0 Vue filter (primary mechanism)
- `widget.computedHeight = 4` on hide; `_origComputedHeight` saved and restored on show (fallback: `32`) — Nodes 2.0 Vue CSS height control
- `node.widgets.splice(0, 0)` unconditionally — triggers Vue reactive proxy re-evaluation of `options.hidden`
- Saves/restores the original `draw()` function to prevent text-bleeding on STRING widgets even when hidden
- `isTile` / `isVideo` value checks extended to include integer `1` (Nodes 2.0 stores toggle values as `0`/`1`)

#### `getTrackedState()` function
- Safely extracts key widget values for state tracking (steps, cfg, sampler, scheduler, denoise, flux_shift, flux_guidance, force_exact_steps, terminal_sigma_to_zero)

#### `toggleDynamicFields()` function
- Unified visibility controller for `cond_weight_b` (hidden when multi_cond_mode=Off) and tile sub-widgets (`tile_size`, `tile_overlap`, `tile_blend` hidden when `tile_mode=false`)

#### `checkSigmaConnection()` function — real-time bypass/mute monitoring
- Checks if `sigmas_override` input has an active cable (not muted/bypassed upstream)
- Mode 2 = Muted, Mode 4 = Bypassed — both treated as "not connected"
- Disables: `steps`, `denoise`, `scheduler`, `flux_shift`, `scheduler_mode`, `start_step`, `end_step`, `terminal_sigma_to_zero`, `ays_schedule` when sigmas_override is active

#### `onDrawBackground` hook
- Uses `onDrawBackground` instead of `onConnectionsChange` for continuous real-time bypass/mute detection (checks every canvas redraw)

#### Preset callback rewrite
- Deduplication: only fires logic when value actually changes (`lastPresetValue` tracking)
- Fixed: `"None (Custom)"` now correctly unlocks all widgets (previous version skipped `updateUILocks` for custom)
- `applyPreset()`: values are applied silently (no callback loops)

---

### `nodes_io.py`

#### Audio export: new `write_external_audio_file` dropdown
- Replaces the old boolean `extract_audio_wav`
- Options: `"None"`, `"WAV — PCM 32-bit Float"`, `"WAV — PCM 24-bit"`, `"WAV — PCM 16-bit"`, `"AIFF — PCM 24-bit"`, `"FLAC — Lossless"`
- WAV 32-bit Float: pure Python implementation (no ffmpeg dependency)
- All other formats: ffmpeg subprocess (raw f32le pipe → target codec)

#### New `audio_filename_suffix` widget
- Default `"_audio"` — appended to `filename_prefix` for the exported audio file

#### Audio export directory fix
- `_write_audio_file()` is now called **after** image writes so the sequence subfolder (`prefix_timestamp/`) is resolved first
- In Sequence mode: audio lands inside the frame subfolder alongside EXR/PNG files (not the parent)

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
- Added `AUDIO_EXPORT_FORMATS` constant

#### H.264 encoder fix
- Switched from `imageio` (no quality control) to `ffmpeg` subprocess with `-crf`
- `quality 0–100` maps to CRF `51–0` (mirroring existing H.265 path)
- Uses `-preset slow -movflags +faststart` for web-compatible output

#### JPEG quality scale fix
- Removed erroneous `×10` multiplier — `cv2.IMWRITE_JPEG_QUALITY` expects `0–100` directly; values above 10 previously all clamped to maximum quality

#### Quality default raised
- Default: `10` → `80` (CRF ~10 for H.264/H.265, 80/100 for JPEG/WEBP)

#### EXR loader error handling
- Split bare `except: pass` into `except ImportError: pass` + `except Exception as e: logger.warning(...)` — real read failures now appear in the log

#### Detailed widget tooltips added
- `quality`: documents CRF mapping for H.264/H.265, direct scale for JPEG/WEBP, no-effect for ProRes/PNG/EXR/HDR
- `output_color_space`: describes each color space (sRGB, Linear, ACEScg, log variants)
- `broadcast_safe`: clarifies it only applies to sRGB mode; `False` required for HDR/VFX pipelines
- `output_path`: documents relative vs absolute path behavior, Windows drive-root behavior for leading `/`
- `compression`: describes all EXR compression modes (ZIP, ZIPS, PIZ, RLE, None, PXR24, B44, DWAA, DWAB)
- `alpha_mode`: documents options; explicitly notes this setting is **not yet functional** (RGB-only output)
- `custom_metadata`: documents key=value format, per-container behavior (EXR header attributes, PNG tEXt chunks)

---

### `js/radiance_io.js`

#### `FORMAT_GROUPS` constant
```js
{ "Video": [...], "Sequence": [...], "Single Image": [...] }
```
Mirrors `WRITE_FORMATS` from `nodes_io.py`. GIF/WEBP placed correctly per mode.

#### `setWidgetVisible()` function (full implementation)
Four-mechanism visibility approach supporting both LiteGraph canvas and Nodes 2.0:
1. `widget.options.hidden = !visible` — Nodes 2.0 Vue filter (primary mechanism)
2. `type="hidden"` + `computeSize=[0,-4]` — LiteGraph canvas: collapses height for all widget types
3. `widget.draw = () => {}` — prevents text bleeding on STRING widgets
4. `inputEl/element display:none` — hides DOM node for customtext widgets
- `widget.computedHeight = 4` on hide; `_origComputedHeight` saved and restored on show (fallback: `32`) — Nodes 2.0 Vue CSS height control
- `node.widgets.splice(0, 0)` unconditionally at end of function — triggers Vue reactive proxy re-evaluation
- On restore: uses `delete widget.computeSize` (not a saved value) to avoid "tiny scrollbar" bug when ComfyUI hasn't yet initialized `computeSize` at first hide

#### `refreshNodeSize()` function
- Forces exact height (grow AND shrink) after widget show/hide so the node resizes correctly

#### Dynamic format filtering
- `formatWidget.options.values` updated on every `write_mode` change
- If current format is no longer valid for the new mode, resets to first valid option

#### Dynamic quality label
- `"quality (CRF)"` — H.264 / H.265
- `"quality (0–100)"` — JPEG / WEBP
- `"quality"` — all others

#### Quality widget visibility
- Hidden for formats that ignore it: EXR, PNG, ProRes, Radiance HDR
- Visible only for: H.264, H.265, JPEG, WEBP

#### Audio suffix visibility
- `audio_filename_suffix` hidden when `write_external_audio_file = "None"`

---

### `nodes_depth.py`

#### VRAM / OOM fix for Depth Anything V2
- Added per-frame GPU tensor release: `del outputs`, `del inputs`, `.cpu()` calls inside the depth estimation loop
- Prevents Out-of-Memory errors when running depth estimation alongside LTX 2.3 (~23 GB VRAM total)

---

### `hdr/vae.py`

#### New inputs: `temporal_size` and `temporal_overlap`
- Added to both `RadianceVAE4KDecode` and the intermediate `_tiled_decode()` method
- `temporal_size` (INT, default 64): chunk size for 3D VAE temporal tiling; `0` = disable
- `temporal_overlap` (INT, default 4): overlap between temporal chunks to prevent flickering

#### New `_temporal_decode()` method
- Decodes a 5D latent chunk temporally with cosine blending across overlapping frame regions
- Prevents VRAM OOM and flickering on long videos with LTX 2.3's 3D VAE
- Blending uses `sin²/cos²` ramps on overlap zones

#### Temporal decoding routing
- Non-tiled branch: if latent is 5D and is a 3D VAE, routes through `_temporal_decode()` instead of direct `vae.decode()`
- Tiled branch: `temporal_size` and `temporal_overlap` passed through to `_tiled_decode()` recursion

#### New input: `crop_bbox` (`BOUNDING_BOX`, `forceInput`)
- Accepts the `crop_bbox` output from `RadianceResolution`
- When connected, crops the decoded image to the broadcast resolution (e.g. 1920×1088 → 1920×1080) directly inside the decode node
- Replaces the need for an external `ImageCropV2` node downstream

---

### `workflows/` *(new files)*

- `LTX 2.3 (Test Workflow).rad` — Single-pass LTX 2.3 generation workflow
- `LTX 2.3 (Two 2x Latent Upscales Test Workflow).rad` — LTX 2.3 with two successive 2× latent upscale passes

---

## Commit History

| Hash | Description |
|------|-------------|
| *(pending)* | fix(js): Nodes 2.0 widget visibility — options.hidden, computedHeight, deferred init |
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
- `alpha_mode` is **not yet functional** — all modes output RGB only. The infrastructure and tooltip are in place for future implementation.
- The new `.rad` workflows are test files; they can be excluded from the main release if preferred.
- The `"Baked VAE (from UNET)"` default for the VAE widget is a behavior change: users who previously connected an explicit VAE file will need to re-select it.
