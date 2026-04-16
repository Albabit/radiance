import json
import logging
import math
import os
import time

import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES

import urllib.request
import tqdm

logger = logging.getLogger("◎ Radiance.loader")

# ═══════════════════════════════════════════════════════════════════════════════
#                         RADIANCE MODEL RESOURCE MAP
# ═══════════════════════════════════════════════════════════════════════════════

RADIANCE_MODEL_MAP = {
    # Diffusion Models (UNET / DiT)
    "flux1-schnell-fp8.safetensors": {
        "url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-schnell-fp8.safetensors",
        "type": "diffusion_models"
    },
    "flux1-dev-fp8.safetensors": {
        "url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors",
        "type": "diffusion_models"
    },
    "sd_xl_base_1.0.safetensors": {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "type": "diffusion_models"
    },
    # Text Encoders (CLIP / T5)
    "t5xxl_fp8_e4m3fn.safetensors": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        "type": "text_encoders"
    },
    "clip_l.safetensors": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "type": "text_encoders"
    },
    # VAE
    "ae.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
        "type": "vae"
    },
    # LTX 2.3 (ALBABIT-FIX)
    "ltx-2.3-22b-dev.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-dev.safetensors",
        "type": "diffusion_models"
    },
    "ltx-2.3-22b-dev-fp8.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.3-fp8/blob/main/ltx-2.3-22b-dev-fp8.safetensors",
        "type": "diffusion_models"
    },
    "gemma_3_12B_it.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/ltx-2/blob/main/split_files/text_encoders/gemma_3_12B_it.safetensors",
        "type": "text_encoders"
    },
    "gemma_3_12B_it_fp4_mixed.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/ltx-2/blob/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        "type": "text_encoders"
    },
    "ltx-2.3_text_projection_bf16.safetensors": {
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/blob/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        "type": "text_encoders"
    },
    "LTX23_video_vae_bf16.safetensors": {
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/blob/main/vae/LTX23_video_vae_bf16.safetensors",
        "type": "vae"
    },
    "LTX23_audio_vae_bf16.safetensors": {
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/blob/main/vae/LTX23_audio_vae_bf16.safetensors",
        "type": "vae"
    }
}


def _download_model(url: str, target_path: str):
    """Download a model with a progress bar in the terminal."""
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        logger.info(f"📥 Radiance: Downloading model from {url}...")
        
        def progress_bar(t):
            last_b = [0]
            def update_to(b=1, bsize=1, tsize=None):
                if tsize is not None: t.total = tsize
                t.update((b - last_b[0]) * bsize)
                last_b[0] = b
            return update_to

        with tqdm.tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(target_path)) as t:
            urllib.request.urlretrieve(url, filename=target_path, reporthook=progress_bar(t))
        
        logger.info(f"✅ Download complete: {target_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Download failed for {url}: {e}")
        return False


def _ensure_model_exists(name: str, folder_type: str, auto_download: bool = False) -> str | None:
    """Check if model exists, download if missing and auto_download is enabled."""
    if not name or name == "None":
        return None
        
    path = folder_paths.get_full_path(folder_type, name)
    if path and os.path.exists(path):
        return path
        
    if auto_download and name in RADIANCE_MODEL_MAP:
        res = RADIANCE_MODEL_MAP[name]
        if res["type"] == folder_type:
            # Construct target path if it doesn't exist
            # We use the first valid folder path for the type
            base_dir = folder_paths.get_folder_paths(folder_type)[0]
            target_path = os.path.join(base_dir, name)
            
            if _download_model(res["url"], target_path):
                # Re-scan to update ComfyUI's internal list
                folder_paths.get_filename_list(folder_type) 
                return target_path
                
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#                       ARCHITECTURE AUTO-DETECTION  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

# Keys sampled from the first N keys of the state dict.
# Order matters — more specific patterns must come first.
#
# v3.1 FIX (BUG-DETECT-1): LTX heuristic (`patch_embedding.weight`) was at
# position 4, BEFORE the Wan heuristic at position 6. Since Wan 2.1 also has
# `patch_embedding.weight`, ALL Wan models were silently misdetected as LTX.
# Fix: Move Wan before LTX, and make LTX use `patchify_proj` (unique to LTX
# architecture) as the primary key instead of the generic `patch_embedding`.
#
# v3.1 ADD: LTX-Video 2.3 detection via `patchify_proj` and `scale_shift_table`
# keys specific to the LTXV 0.9.5+ architecture.
_ARCH_HEURISTICS = [
    # ── High-confidence unique keys (order doesn't matter among these) ──

    # Flux: unique double-stream block naming
    (lambda ks: any("double_blocks" in k for k in ks),             "flux"),
    # SD3 / SD3.5: joint transformer blocks
    (lambda ks: any("joint_blocks" in k for k in ks),              "sd3"),
    # HunyuanVideo: image input projection
    (lambda ks: any("img_in" in k and "proj" in k for k in ks),    "hunyuan_video"),
    # Lumina / Z-Image: unique caption projection key
    (lambda ks: any("cap_v_projection.weight" in k for k in ks),   "lumina2"),
    # Kolors: Chatglm style conditioning
    (lambda ks: any("chatglm" in k.lower() for k in ks),           "kolors"),
    # AuraFlow: unique prefix
    (lambda ks: any("auraflow" in k.lower() for k in ks),          "aura_flow"),

    # ── Requires ordering: these share overlapping keys ──────────────

    # Wan 2.1: patch_embedding + time_embedding (no joint_blocks).
    # MUST come before LTX — both can have patch_embedding, but only Wan
    # has time_embedding in the first 200 keys.
    # FIX 5: Previous heuristic checked for "wan" literal in first 5 key names
    # which fails for all Wan 2.1 checkpoints (keys start with
    # "model.diffusion_model.patch_embedding..." — no "wan" substring).
    (lambda ks: any("patch_embedding" in k for k in ks)
              and any("time_embedding" in k for k in ks)
              and not any("joint_blocks" in k for k in ks),         "wan"),

    # LTX-Video (all versions including 2.3): patchify_proj is unique to LTXV
    # architecture and NOT present in Wan, SD3, Flux, or PixArt models.
    # v3.1: Primary key is `patchify_proj`; fallback checks for `patch_embedding`
    # + `adaln_single` + NO `time_embedding` (excludes Wan).
    (lambda ks: any("patchify_proj" in k for k in ks),             "ltx"),
    # LTX fallback: older LTXV checkpoints that use patch_embedding naming
    (lambda ks: any("patch_embedding" in k for k in ks)
              and any("adaln_single" in k for k in ks)
              and not any("time_embedding" in k for k in ks),       "ltx"),

    # PixArt: adaln_single WITHOUT patchify_proj (LTX also has adaln_single)
    # v3.1 FIX: Added exclusion of `patchify_proj` to prevent PixArt matching LTX
    (lambda ks: any("adaln_single" in k for k in ks)
              and not any("patchify_proj" in k for k in ks),        "pixart"),

    # ── UNet-based models (least specific — always last) ─────────────

    # SDXL: has both UNet input_blocks AND add_embedding (refiner clue)
    (lambda ks: any("down_blocks.0" in k for k in ks)
              and any("add_embedding" in k for k in ks),            "sdxl"),
    # SD1.5: UNet with input_blocks naming
    (lambda ks: any("input_blocks.0" in k for k in ks),            "sd1.5"),
    # SDXL fallback: down_blocks without add_embedding → still SDXL
    (lambda ks: any("down_blocks.0" in k for k in ks),             "sdxl"),
]

# v3.1: Increased from 80 → 200. Some model architectures (especially Wan 2.1
# and LTX 2.3) have their differentiating keys beyond the first 80 entries.
# The extra scan time (~1ms) is negligible compared to the multi-second model load.
_SAFETENSORS_PEEK = 200


def _detect_model_type(unet_path: str) -> str | None:
    """
    Detect model architecture from safetensors metadata (key heuristics).
    Returns a model_type string or None if detection failed.
    Only reads the first _SAFETENSORS_PEEK keys for speed.
    """
    try:
        from safetensors import safe_open
        with safe_open(unet_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())[:_SAFETENSORS_PEEK]
        for test_fn, arch in _ARCH_HEURISTICS:
            if test_fn(keys):
                logger.info(f"🔍 Auto-detected architecture: {arch} from {os.path.basename(unet_path)}")
                return arch
    except ImportError:
        logger.debug("safetensors not available — skipping auto-detect")
    except Exception as e:
        logger.debug(f"Auto-detect failed: {e}")
    return None


def _file_fingerprint(path: str) -> str:
    """Return a cache key suffix that changes when the file changes."""
    try:
        st = os.stat(path)
        return f"{st.st_mtime:.0f}:{st.st_size}"
    except OSError:
        return "nostat"


# ═══════════════════════════════════════════════════════════════════════════════
#                         LATENT FORMAT TABLE  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

LATENT_CHANNELS = {
    "flux":           16,
    "sd3":            16,
    "sd3.5":          16,
    "ltx":            16,
    "ltxav":          16, # ALBABIT-FIX: ensure LTX 2.3 gets 16 channels mapping
    "hunyuan_video":  16,
    "wan":            16,
    "lumina2":        16,
    "z_image":        16,
    "sdxl":            4,
    "sd1.5":           4,
    "pixart":          4,
    "aura_flow":       4,
    "kolors":          4,
}


def _latent_format(arch: str) -> str:
    """Return a latent format label compatible with Radiance Sampler Pro.

    v3.1 FIX (BUG-FORMAT-1): Previously returned bare channel count like "16ch"
    which didn't match the format labels in vae.py's LATENT_FORMAT_MAP (e.g.
    "flux_16ch", "sd_4ch"). This caused format mismatches when the Sampler Pro
    tried to look up the format from the loader's output string.

    Now returns architecture-prefixed labels: "flux_16ch", "ltx_16ch", "sd_4ch", etc.
    """
    # Architecture → format label mapping
    # Must match the labels that vae.py and nodes_sampler.py expect.
    _FORMAT_MAP = {
        "flux":           "flux_16ch",
        "sd3":            "sd3_16ch",
        "sd3.5":          "sd3_16ch",
        "ltx":            "ltx_16ch",
        "ltxav":          "ltx_16ch", # ALBABIT-FIX: Format label
        "hunyuan_video":  "hunyuan_16ch",
        "wan":            "wan_16ch",
        "lumina2":        "lumina_16ch",
        "z_image":        "z_image_16ch",
        "sdxl":           "sd_4ch",
        "sd1.5":          "sd_4ch",
        "pixart":         "sd_4ch",
        "aura_flow":      "sd_4ch",
        "kolors":         "sd_4ch",
    }
    return _FORMAT_MAP.get(arch, f"{arch}_{LATENT_CHANNELS.get(arch, 4)}ch")


# ═══════════════════════════════════════════════════════════════════════════════
#                       CLIP ASSEMBLY RULES  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

# Defines which named CLIP slots to use per architecture, in load order.
# Slots: "clip_l" | "clip_g" | "t5xxl" | "llm_encoder" | "text_projection"
CLIP_SLOT_ORDER = {
    "flux":           ["clip_l", "t5xxl"],
    "sd3":            ["clip_l", "clip_g", "t5xxl"],
    "sd3.5":          ["clip_l", "clip_g", "t5xxl"],
    "sdxl":           ["clip_l", "clip_g"],
    "sd1.5":          ["clip_l"],
    "hunyuan_video":  ["llm_encoder", "clip_l"],
    "wan":            ["t5xxl"],
    "ltx":            ["llm_encoder", "text_projection"], # ALBABIT-FIX: Reassigned from t5xxl to llm_encoder
    "ltxav":          ["llm_encoder", "text_projection"], # ALBABIT-FIX: Added specific type
    "lumina2":        ["t5xxl"],
    "z_image":        ["t5xxl"],
    "pixart":         ["t5xxl"],
    "aura_flow":      ["clip_l"],
    "kolors":         ["llm_encoder"],
}


def _assemble_clip_paths(arch: str, clip_l, clip_g, t5xxl, llm_encoder, text_projection) -> list[str]: # ALBABIT-FIX: Added text_projection
    """
    Build ordered list of CLIP paths from named slots for the given architecture.
    Only includes slots that are filled (not None/empty string).
    Falls back to any non-empty slot if arch is unknown.
    """
    slot_map = {
        "clip_l":          clip_l,
        "clip_g":          clip_g,
        "t5xxl":           t5xxl,
        "llm_encoder":     llm_encoder,
        "text_projection": text_projection, # ALBABIT-FIX
    }
    order = CLIP_SLOT_ORDER.get(arch, list(slot_map.keys()))
    paths = []
    for slot in order:
        val = slot_map.get(slot)
        if val and val not in ("None", ""):
            p = folder_paths.get_full_path("text_encoders", val)
            if p:
                paths.append(p)
            else:
                logger.warning(f"◎ CLIP slot '{slot}' file not found: {val}")
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
#                         CHECKPOINT PRESETS  (v3.0 — updated)
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PRESETS = {
    "None (Manual)": {},
    # ── Flux ──
    "→ Flux Dev": {
        "model_type":    "flux",
        "weight_dtype":  "fp8_e4m3fn",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "t5xxl": True},
        "vram_gb":       12,
    },
    "→ Flux Schnell": {
        "model_type":    "flux",
        "weight_dtype":  "fp8_e4m3fn",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "t5xxl": True},
        "vram_gb":       10,
    },
    "→ Flux Dev (Low VRAM)": {
        "model_type":    "flux",
        "weight_dtype":  "fp8_e4m3fn",
        "clip_dtype":    "fp8_e4m3fn",
        "offload_mode":  "cpu_offload",
        "clip_slots":    {"clip_l": True, "t5xxl": True},
        "vram_gb":       8,
    },
    # ── SD3.x ──
    "→ SD3.5 Large": {
        "model_type":    "sd3.5",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "clip_g": True, "t5xxl": True},
        "vram_gb":       16,
    },
    "→ SD3.5 Medium": {
        "model_type":    "sd3.5",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "clip_g": True, "t5xxl": True},
        "vram_gb":       10,
    },
    "→ SD3.5 Turbo": {
        "model_type":    "sd3.5",
        "weight_dtype":  "bf16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "clip_g": True, "t5xxl": True},
        "vram_gb":       12,
    },
    # ── SDXL ──
    "→ SDXL Base": {
        "model_type":    "sdxl",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "clip_g": True},
        "vram_gb":       8,
    },
    "→ SDXL Turbo": {
        "model_type":    "sdxl",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "clip_g": True},
        "vram_gb":       6,
    },
    # ── SD 1.5 ──
    "→ SD 1.5": {
        "model_type":    "sd1.5",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True},
        "vram_gb":       4,
    },
    # ── Video Models ──
    "→ HunyuanVideo": {
        "model_type":    "hunyuan_video",
        "weight_dtype":  "fp8_e4m3fn",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"llm_encoder": True, "clip_l": True},
        "vram_gb":       24,
    },
    "→ Wan 2.1": {
        "model_type":    "wan",
        "weight_dtype":  "fp8_e4m3fn",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"t5xxl": True},
        "vram_gb":       16,
    },
    "→ LTX Video": {
        "model_type":    "ltx",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"t5xxl": True},
        "vram_gb":       12,
    },
    # ALBABIT-FIX: Reordered 13B preset to be below base LTX
    "→ LTX Video 13B": {
        "model_type":    "ltx",
        "weight_dtype":  "fp8_e4m3fn",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"t5xxl": True},
        "vram_gb":       18,
    },
    "→ LTX Video 2.3": { # ALBABIT-FIX: Specific LTX 2.3 preset with ltxav
        "model_type":    "ltxav",
        "weight_dtype":  "bf16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"llm_encoder": True},
        "vram_gb":       14,
    },
    "→ LTX Video 2.3 (Low VRAM)": { # ALBABIT-FIX: specific LTX 2.3 low vram with ltxav
        "model_type":    "ltxav",
        "weight_dtype":  "fp8_e4m3fn",
        "clip_dtype":    "fp8_e4m3fn",
        "offload_mode":  "cpu_offload",
        "clip_slots":    {"llm_encoder": True},
        "vram_gb":       8,
    },
    # ── Other Image Models ──
    "→ PixArt Sigma": {
        "model_type":    "pixart",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"t5xxl": True},
        "vram_gb":       8,
    },
    "→ AuraFlow": {
        "model_type":    "aura_flow",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True},
        "vram_gb":       10,
    },
    "→ Kolors": {
        "model_type":    "kolors",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"llm_encoder": True},
        "vram_gb":       10,
    },
    "→ Lumina2": {
        "model_type":    "lumina2",
        "weight_dtype":  "bf16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"t5xxl": True},
        "vram_gb":       14,
    },
    "→ Z-Image": {
        "model_type":    "z_image",
        "weight_dtype":  "bf16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"t5xxl": True},
        "vram_gb":       16,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#                         VRAM UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_vram_usage(
    model_type: str,
    weight_dtype: str,
    clip_dtype: str = "fp16",
    has_loras: bool = False,
    has_controlnet: bool = False,
) -> float:
    base_vram = {
        "flux": 12.0, "sd3": 10.0, "sd3.5": 12.0,
        "sdxl": 6.5, "sd1.5": 3.5,
        "hunyuan_video": 20.0, "wan": 14.0, "ltx": 11.0, "ltxav": 15.0, # ALBABIT-FIX
        "pixart": 6.0, "aura_flow": 8.0, "kolors": 8.0,
        "lumina2": 12.0, "z_image": 14.0,
    }.get(model_type, 8.0)

    unet_mult = {
        "fp32": 2.0, "fp16": 1.0, "bf16": 1.0,
        "fp8_e4m3fn": 0.6, "fp8_e5m2": 0.6, "default": 1.0,
    }.get(weight_dtype, 1.0)

    clip_vram = {
        "flux": 4.5, "sd3": 3.0, "sd3.5": 3.5,
        "sdxl": 1.5, "sd1.5": 0.8,
        "hunyuan_video": 4.5, "wan": 3.0, "ltx": 2.5, "ltxav": 8.0, # ALBABIT-FIX
        "pixart": 2.0, "aura_flow": 2.0, "kolors": 3.0,
        "lumina2": 3.0, "z_image": 3.0,
    }.get(model_type, 2.0)

    clip_mult = {
        "fp32": 2.0, "fp16": 1.0, "bf16": 1.0,
        "fp8_e4m3fn": 0.55, "fp8_e5m2": 0.55, "default": 1.0,
    }.get(clip_dtype, 1.0)

    vram = (base_vram * unet_mult) + (clip_vram * clip_mult)
    if has_loras:
        vram += 0.5
    if has_controlnet:
        vram += 2.0
    return round(vram, 1)


def get_available_vram() -> float:
    try:
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info(0)
            return round(free_mem / (1024 ** 3), 1)
    except Exception:  # nosec B110
        pass
    return 0.0


def get_total_vram() -> float:
    try:
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            return round(total_mem / (1024 ** 3), 1)
    except Exception:  # nosec B110
        pass
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#                        CLIP TYPE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def get_clip_type_enum(model_type: str):
    """Resolve ComfyUI CLIPType enum for a given model architecture.

    v3.1 FIX (BUG-CLIP-1): The variant search for "ltx" tried "LTX", "LTX",
    "Ltx" — none of which match ComfyUI's actual enum name "LTX_VIDEO" or
    "LTXV". Fixed by adding explicit composite name variants for models whose
    CLIPType name doesn't match a simple uppercase of the model_type string.
    """
    mapping = {
        "flux":   comfy.sd.CLIPType.FLUX,
        "sd3":    comfy.sd.CLIPType.SD3,
        "sd3.5":  comfy.sd.CLIPType.SD3,
        "sdxl":   comfy.sd.CLIPType.STABLE_DIFFUSION,
        "sd1.5":  comfy.sd.CLIPType.STABLE_DIFFUSION,
    }

    # v3.1: Extended variant list per model — handles cases where the ComfyUI
    # CLIPType enum name doesn't match a simple uppercase of the model_type.
    # E.g. model_type "ltx" → CLIPType.LTX_VIDEO (not CLIPType.LTX)
    _EXTRA_VARIANTS = {
        "ltx":            ["LTX_VIDEO", "LTXV", "LTX"],
        "ltxav":          ["LTX_VIDEO", "LTXV", "LTX"], # ALBABIT-FIX
        "hunyuan_video":  ["HUNYUAN_VIDEO", "HUNYUANVIDEO"],
        "wan":            ["WAN", "WAN2", "WAN_VIDEO"],
        "aura_flow":      ["AURA_FLOW", "AURAFLOW"],
    }

    for name in ("hunyuan_video", "wan", "ltx", "ltxav", "pixart", "aura_flow", "kolors", "lumina2", "z_image"): # ALBABIT-FIX: ltxav
        # Build variant list: explicit extras first, then the auto-generated names
        enum_name = name.upper().replace(".", "_")
        auto_variants = [enum_name, name.upper(), name.title().replace("_", "")]
        extra = _EXTRA_VARIANTS.get(name, [])
        all_variants = extra + [v for v in auto_variants if v not in extra]

        for variant in all_variants:
            if hasattr(comfy.sd.CLIPType, variant):
                mapping[name] = getattr(comfy.sd.CLIPType, variant)
                break
        else:
            mapping.setdefault(name, comfy.sd.CLIPType.STABLE_DIFFUSION)

    clip_type = mapping.get(model_type)
    if clip_type is None:
        enum_name = model_type.upper().replace(".", "_")
        if hasattr(comfy.sd.CLIPType, enum_name):
            clip_type = getattr(comfy.sd.CLIPType, enum_name)
        else:
            logger.warning(
                f"◎ No CLIPType mapping for '{model_type}', "
                f"falling back to STABLE_DIFFUSION"
            )
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

    return clip_type


# ═══════════════════════════════════════════════════════════════════════════════
#                        MODEL CACHE (LRU)
# ═══════════════════════════════════════════════════════════════════════════════

class _LRUCache:
    """
    Least-Recently-Used model cache using OrderedDict for O(1) hit/evict.

    Keys include mtime+size fingerprint so stale entries are automatically
    missed when files change on disk — no manual invalidation needed.

    FIX 3: Previous implementation used list.remove() on every get() and put()
    which is O(n) — scans the entire access list on every cache hit.
    Upgraded to collections.OrderedDict + move_to_end() for O(1) LRU,
    consistent with the SigmaCache upgrade in nodes_sampler.py.
    """

    def __init__(self, max_size: int = 4):
        from collections import OrderedDict
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self._max_size = max_size

    def get(self, key: str):
        if key in self._cache:
            self._cache.move_to_end(key)   # O(1) — mark as recently used
            return self._cache[key]
        return None

    def put(self, key: str, obj) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                evicted, _ = self._cache.popitem(last=False)  # O(1) LRU evict
                evicted_name = evicted.split(":")[1] if ":" in evicted else evicted
                logger.info(f"Cache evicted: {evicted_name}")
        self._cache[key] = obj

    def has(self, key: str) -> bool:
        return key in self._cache

    def clear(self) -> None:
        self._cache.clear()
        logger.info("Model cache cleared")

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def size(self) -> int:
        return len(self._cache)


# v3.1: Reduce default cache size to 2 to prevent VRAM exhaustion with Flux/SD3.5.
# Users can override via RADIANCE_CACHE_SIZE env var.
# FIX 2: Previous env var "◎ Radiance_CACHE_SIZE" was impossible to set from
# any shell — it contained a Unicode ◎ character and a space. Renamed to the
# standard POSIX-safe name. Set via: export RADIANCE_CACHE_SIZE=4
_DEFAULT_CACHE_SIZE = int(os.environ.get("RADIANCE_CACHE_SIZE", 2))
_cache = _LRUCache(max_size=_DEFAULT_CACHE_SIZE)


# ═══════════════════════════════════════════════════════════════════════════════
#                     RADIANCE LORA STACK NODE  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceLoraStack:
    """
    Compose up to 5 LoRAs into a LORA_STACK for use with RadianceUnifiedLoader.
    Can accept an upstream LORA_STACK to chain stacks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # ALBABIT-FIX: Safe `or []` to prevent initialization crash on empty folders
        lora_list = ["None"] + (folder_paths.get_filename_list("loras") or [])
        lora_slot = lambda tooltip: (
            lora_list,
            {"default": "None", "tooltip": tooltip},
        )
        str_slot = lambda tooltip: (
            "FLOAT",
            {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
             "tooltip": tooltip},
        )
        return {
            "required": {},
            "optional": {
                "lora_stack":     ("LORA_STACK", {"default": None,
                    "tooltip": "Chain an upstream LORA_STACK before these LoRAs."}),
                "lora_1":         lora_slot("LoRA 1"),
                "lora_1_model":   str_slot("LoRA 1 model strength"),
                "lora_1_clip":    str_slot("LoRA 1 CLIP strength"),
                "lora_2":         lora_slot("LoRA 2"),
                "lora_2_model":   str_slot("LoRA 2 model strength"),
                "lora_2_clip":    str_slot("LoRA 2 CLIP strength"),
                "lora_3":         lora_slot("LoRA 3"),
                "lora_3_model":   str_slot("LoRA 3 model strength"),
                "lora_3_clip":    str_slot("LoRA 3 CLIP strength"),
                "lora_4":         lora_slot("LoRA 4"),
                "lora_4_model":   str_slot("LoRA 4 model strength"),
                "lora_4_clip":    str_slot("LoRA 4 CLIP strength"),
                "lora_5":         lora_slot("LoRA 5"),
                "lora_5_model":   str_slot("LoRA 5 model strength"),
                "lora_5_clip":    str_slot("LoRA 5 CLIP strength"),
            },
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "build_stack"
    CATEGORY = "FXTD Studios/Radiance/Generate"
    DESCRIPTION = (
        "Compose up to 5 LoRAs into an accumulating LORA_STACK. "
        "Chain multiple stacks together. Feed into Radiance Unified Loader."
    )

    def build_stack(
        self,
        lora_stack=None,
        lora_1="None", lora_1_model=1.0, lora_1_clip=1.0,
        lora_2="None", lora_2_model=1.0, lora_2_clip=1.0,
        lora_3="None", lora_3_model=1.0, lora_3_clip=1.0,
        lora_4="None", lora_4_model=1.0, lora_4_clip=1.0,
        lora_5="None", lora_5_model=1.0, lora_5_clip=1.0,
    ) -> tuple:
        stack = list(lora_stack) if lora_stack else []
        for name, ms, cs in [
            (lora_1, lora_1_model, lora_1_clip),
            (lora_2, lora_2_model, lora_2_clip),
            (lora_3, lora_3_model, lora_3_clip),
            (lora_4, lora_4_model, lora_4_clip),
            (lora_5, lora_5_model, lora_5_clip),
        ]:
            if name and name != "None":
                stack.append((name, float(ms), float(cs)))
        return (stack,)


# ═══════════════════════════════════════════════════════════════════════════════
#                     RADIANCE UNIFIED LOADER v2.1.0
# ═══════════════════════════════════════════════════════════════════════════════

# ALBABIT-FIX: Added ltxav to valid model types
MODEL_TYPES = [
    "Auto-Detect",
    "flux", "sd3", "sd3.5",
    "sdxl", "sd1.5",
    "hunyuan_video", "wan", "ltx", "ltxav",
    "lumina2", "z_image",
    "pixart", "aura_flow", "kolors",
]

WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"]
CLIP_DTYPES   = ["default", "fp16", "bf16", "fp8_e4m3fn", "fp32"]
OFFLOAD_MODES = ["none", "cpu_offload", "sequential"]


class RadianceUnifiedLoader:
    """
    Universal diffusion model loader v3.1.

    Outputs: MODEL, CLIP, VAE, AUDIO_VAE, CONTROL_NET, LORA_STACK, LATENT_UPSCALE_MODEL,
             load_info (human string), latent_format ("flux_16ch"/"sd_4ch"/etc.),
             model_meta (JSON string with full metadata dict).

    v3.1 fixes: LTX/Wan auto-detect ordering, CLIPType resolution for LTX_VIDEO,
    architecture-prefixed latent_format labels, LTX 2.3 presets.
    
    ALBABIT-FIX: Enhanced with Built-in & Standalone Audio VAE extraction, 
    proper Native ComfyUI standalone Video VAE routing to prevent size mismatch, 
    native Gemma 3 / text_projection loading capabilities, and Latent Upscale Model loader.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # ALBABIT-FIX: Safe `or []` lists to prevent python type addition crashes
        lora_list    = ["None"] + (folder_paths.get_filename_list("loras") or [])
        clip_list    = ["None"] + (folder_paths.get_filename_list("text_encoders") or [])
        cn_list      = ["None"] + (folder_paths.get_filename_list("controlnet") or [])
        upscale_list = ["None"] + (folder_paths.get_filename_list("latent_upscale_models") or []) # ALBABIT-FIX

        # ALBABIT-FIX: Safe dynamic lists for Audio VAE & Baked VAE support
        ckpt_files = folder_paths.get_filename_list("checkpoints") or []
        vae_files = folder_paths.get_filename_list("vae") or []
        audio_vae_list = ["None", "Baked Audio VAE (from UNET)"] + sorted(list(set(ckpt_files + vae_files)))
        vae_list = ["Baked VAE (from UNET)"] + vae_files

        clip_slot = lambda tip: (clip_list, {"default": "None", "tooltip": tip})

        return {
            "required": {
                # ── Preset ──
                "preset": (
                    list(CHECKPOINT_PRESETS.keys()),
                    {"default": "None (Manual)",
                     "tooltip": "Quick-configure for common architectures. "
                                "ALBABIT-FIX: Only overrides model_type now to respect user's dtype choices."},
                ),
                # ── UNET ──
                "unet_name": (
                    folder_paths.get_filename_list("diffusion_models") or [],
                    {"tooltip": "Main diffusion model (UNET / DiT / Transformer)."},
                ),
                "weight_dtype": (
                    WEIGHT_DTYPES,
                    {"default": "default",
                     "tooltip": "UNET weight precision. fp8_e4m3fn saves ~40% VRAM vs fp16."},
                ),
                # ── Architecture ──
                "model_type": (
                    MODEL_TYPES,
                    {"default": "Auto-Detect",
                     "tooltip": "'Auto-Detect' reads the checkpoint's key names to determine "
                                "architecture. Override manually if detection fails."},
                ),
                # ── VAE ──
                "vae_name": (
                    vae_list,
                    {"tooltip": "VAE for encoding/decoding latents.", "default": "Baked VAE (from UNET)"},
                ),
            },
            "optional": {
                # ── AUDIO VAE (ALBABIT-FIX) ──
                "audio_vae_name": (
                    audio_vae_list, 
                    {"default": "None", "tooltip": "Audio VAE for LTX 2.3. Choose 'Baked' or a standalone safetensors file."}
                ),
                # ── UPSCALER (ALBABIT-FIX) ──
                "upscale_model_name": (
                    upscale_list, 
                    {"default": "None", "tooltip": "Latent Upscale Model (e.g., for LTX 2.3 or HunyuanVideo)."}
                ),
                # ── Named CLIP slots ──
                "clip_l":      clip_slot(
                    "CLIP-L (text encoder). Used by: SD1.5, SDXL, Flux, SD3."),
                "clip_g":      clip_slot(
                    "CLIP-G (text encoder). Used by: SDXL, SD3, SD3.5."),
                "t5xxl":       clip_slot(
                    "T5-XXL (text encoder). Used by: Flux, SD3, SD3.5, Wan, PixArt."),
                "llm_encoder": clip_slot(
                    "LLM encoder (Gemma 3). Used by: Kolors, HunyuanVideo, LTX 2.3."), # ALBABIT-FIX
                "text_projection": clip_slot(
                    "Text Projection Model. Used with Gemma 3 for native DualCLIP loading in LTX 2.3."), # ALBABIT-FIX
                # ── CLIP precision ──
                "clip_dtype": (
                    CLIP_DTYPES,
                    {"default": "default",
                     "tooltip": "CLIP weight precision. Independent from UNET. "
                                "For Flux T5XXL: fp8 saves ~4.7 GB vs fp16."},
                ),
                # ── Offload ──
                "offload_mode": (
                    OFFLOAD_MODES,
                    {"default": "none",
                     "tooltip": "none = GPU only. "
                                "cpu_offload = CLIP loaded to CPU RAM. "
                                "sequential = enable ComfyUI sequential CPU offload (8–12 GB GPUs)."},
                ),
                # ── LoRA (built-in slots) ──
                "lora_stack":      ("LORA_STACK", {"default": None,
                    "tooltip": "Accept a LORA_STACK from RadianceLoraStack node."}),
                "lora_1":          (lora_list, {"default": "None"}),
                "lora_1_model_str":("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA 1 strength on model."}),
                "lora_1_clip_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA 1 strength on CLIP."}),
                "lora_2":          (lora_list, {"default": "None"}),
                "lora_2_model_str":("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_2_clip_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_3":          (lora_list, {"default": "None"}),
                "lora_3_model_str":("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_3_clip_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                # ── ControlNet ──
                "controlnet_name": (cn_list, {"default": "None",
                    "tooltip": "Optional ControlNet model."}),
                "controlnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "controlnet_start":    ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "controlnet_end":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # ── Options ──
                "check_vram":  (["On", "Off"], {"default": "On",
                    "tooltip": "Estimate VRAM before load and warn if tight."}),
                "use_cache":   (["On", "Off"], {"default": "On",
                    "tooltip": "Cache loaded models. Skips disk I/O when re-running "
                               "with the same files. Cache auto-invalidates if files change."}),
                "lora_on_error": (["warn", "raise"], {"default": "raise",
                    "tooltip": "'warn' skips failed LoRA and continues. "
                               " 'raise' stops execution."}),
                "auto_download": ("BOOLEAN", {"default": False,
                    "tooltip": "If a selected model is missing, automatically download it from Radiance mirrors."}),
            },
        }

    # ALBABIT-FIX: Return AUDIO_VAE & LATENT_UPSCALE_MODEL
    RETURN_TYPES  = ("MODEL", "CLIP", "VAE", "VAE", "CONTROL_NET", "LORA_STACK", "LATENT_UPSCALE_MODEL",
                     "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("MODEL", "CLIP", "VAE", "AUDIO_VAE", "CONTROLNET", "lora_stack", "upscale_model",
                     "load_info", "latent_format", "model_meta")
    FUNCTION      = "load_radiance_stack"
    CATEGORY      = "FXTD Studios/Radiance/Generate"
    DESCRIPTION   = (
        "Universal loader v3.1 — auto-detects architecture (Flux, SD3, SDXL, "
        "Wan, LTX 2.3, HunyuanVideo, PixArt, Lumina2, etc.), named CLIP slots, "
        "chainable LORA_STACK, model_meta JSON, latent_format, offload mode. "
        "ZERO stale cache hits (mtime + size fingerprinting)."
    )

    def load_radiance_stack(
        self,
        preset,
        unet_name,
        weight_dtype,
        model_type,
        vae_name,
        audio_vae_name="None", # ALBABIT-FIX
        upscale_model_name="None", # ALBABIT-FIX
        clip_l="None",
        clip_g="None",
        t5xxl="None",
        llm_encoder="None",
        text_projection="None", # ALBABIT-FIX
        clip_dtype="default",
        offload_mode="none",
        lora_stack=None,
        lora_1="None", lora_1_model_str=1.0, lora_1_clip_str=1.0,
        lora_2="None", lora_2_model_str=1.0, lora_2_clip_str=1.0,
        lora_3="None", lora_3_model_str=1.0, lora_3_clip_str=1.0,
        controlnet_name="None",
        controlnet_strength=1.0,
        controlnet_start=0.0,
        controlnet_end=1.0,
        check_vram="On",
        use_cache="On",
        lora_on_error="raise",
        auto_download=False,
    ):
        load_start  = time.time()
        info_lines  = []
        caching     = use_cache == "On"

        # ════════════════════════════════════════════════════════════════
        # 0. APPLY PRESET
        # ════════════════════════════════════════════════════════════════
        if preset != "None (Manual)" and preset in CHECKPOINT_PRESETS:
            cfg = CHECKPOINT_PRESETS[preset]
            overrides = []

            def _apply(field, key, cur):
                new = cfg.get(key, cur)
                if new != cur:
                    overrides.append(f"{field}: {cur}→{new}")
                return new

            # ALBABIT-FIX: Only override model_type automatically to respect user UI dtypes
            model_type = _apply("model_type", "model_type", model_type)

            msg = (f"✓ Preset '{preset}' applied" +
                   (f" (overrode: {', '.join(overrides)})" if overrides else " (no overrides)") + 
                   ". Dtypes and offload modes strictly follow UI.")
            logger.info(msg)
            info_lines.append(msg)

        # ════════════════════════════════════════════════════════════════
        # 1. RESOLVE ARCHITECTURE
        # ════════════════════════════════════════════════════════════════
        unet_path = _ensure_model_exists(unet_name, "diffusion_models", auto_download)
        if not unet_path:
            raise FileNotFoundError(
                f"❌ UNET not found: '{unet_name}'. Enable auto_download or install it manually."
            )

        detected_type = None
        if model_type == "Auto-Detect":
            detected_type = _detect_model_type(unet_path)
            if detected_type:
                resolved_type = detected_type
                info_lines.append(f"◎ Auto-detected: {resolved_type}")
            else:
                resolved_type = "sdxl"   # safe fallback
                logger.warning(
                    "◎ Architecture auto-detect failed. Falling back to 'sdxl'. "
                    "Set model_type manually if this is wrong."
                )
                info_lines.append("◎ Auto-detect failed — fallback: sdxl")
        else:
            resolved_type = model_type

        latent_fmt  = _latent_format(resolved_type)
        lat_msg     = f"◎ Latent format: {latent_fmt} ({resolved_type})"
        logger.info(lat_msg)
        info_lines.append(lat_msg)

        # ════════════════════════════════════════════════════════════════
        # 2. OFFLOAD MODE
        # ════════════════════════════════════════════════════════════════
        if offload_mode == "sequential":
            try:
                comfy.model_management.set_lowvram_mode(True)
                logger.info("◎ Sequential CPU offload enabled")
                info_lines.append("◎ Offload: sequential")
            except Exception as e:
                logger.warning(f"◎ Could not enable sequential offload: {e}")

        # FIX 6: ComfyUI model_options["load_device"] expects torch.device, not str.
        clip_load_device = torch.device("cpu") if offload_mode == "cpu_offload" else None

        # ════════════════════════════════════════════════════════════════
        # 3. VRAM ESTIMATION
        # ════════════════════════════════════════════════════════════════
        has_loras = (
            any(l != "None" for l in [lora_1, lora_2, lora_3])
            or bool(lora_stack)
        )
        has_cn = bool(controlnet_name and controlnet_name != "None")

        if check_vram == "On":
            est   = estimate_vram_usage(resolved_type, weight_dtype, clip_dtype,
                                        has_loras, has_cn)
            avail = get_available_vram()
            total = get_total_vram()
            vram_msg = (f"◎ VRAM: ~{est} GB needed | "
                        f"{avail} GB free / {total} GB total")
            logger.info(vram_msg)
            info_lines.append(vram_msg)
            if avail > 0 and est > avail * 0.9:
                warn = (f"◎ VRAM tight! {est} GB estimated, {avail} GB free. "
                        f"Consider fp8 dtype or cpu_offload.")
                logger.warning(warn)
                info_lines.append(warn)
        else:
            est = estimate_vram_usage(resolved_type, weight_dtype, clip_dtype,
                                      has_loras, has_cn)

        # ════════════════════════════════════════════════════════════════
        # 4. LOAD UNET & BAKED VAEs (mtime + size cache key)
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        unet_fp   = _file_fingerprint(unet_path)
        unet_key  = f"unet:{unet_path}:{weight_dtype}:{unet_fp}"
        
        # ALBABIT-FIX: Identify dependencies on UNET for baked extraction
        extract_vae = (vae_name == "Baked VAE (from UNET)")
        extract_audio_vae = (audio_vae_name == "Baked Audio VAE (from UNET)")
        baked_vae_key = f"vae:baked:{unet_path}:{unet_fp}"
        baked_audio_vae_key = f"audio_vae:baked:{unet_path}:{unet_fp}"
        
        vae = None
        audio_vae = None

        # FIX 4: Record cache HIT before loading — after _cache.put() the key
        # is always present, so checking has() post-load always returns True.
        unet_cache_hit = caching and _cache.has(unet_key)
        
        # ALBABIT-FIX: Re-evaluate UNET cache hit if we need baked VAEs but they fell out of cache
        if unet_cache_hit:
            if extract_vae and not _cache.has(baked_vae_key):
                unet_cache_hit = False
            if extract_audio_vae and not _cache.has(baked_audio_vae_key):
                unet_cache_hit = False

        if unet_cache_hit:
            model = _cache.get(unet_key)
            logger.info(f"◎ UNET from cache: {unet_name}")
            info_lines.append(f"◎ UNET: {unet_name} (cached)")
            
            if extract_vae:
                vae = _cache.get(baked_vae_key)
                logger.info("◎ BAKED VAE from cache")
                info_lines.append("◎ VAE: Baked from UNET (cached)")
            if extract_audio_vae:
                audio_vae = _cache.get(baked_audio_vae_key)
                logger.info("◎ BAKED AUDIO VAE from cache")
                info_lines.append("◎ AUDIO VAE: Baked from UNET (cached)")
        else:
            model_options = {}
            is_gguf = unet_name.lower().endswith(".gguf")
            if is_gguf:
                logger.info(f"◎ GGUF detected: {unet_name} (embedded quant)")
            else:
                dtype_map = {
                    "fp8_e4m3fn": torch.float8_e4m3fn,
                    "fp8_e5m2":   torch.float8_e5m2,
                    "fp16":       torch.float16,
                    "bf16":       torch.bfloat16,
                    "fp32":       torch.float32,
                }
                if weight_dtype in dtype_map:
                    model_options["dtype"] = dtype_map[weight_dtype]

            try:
                # ALBABIT-FIX: Use load_checkpoint_guess_config if VAEs need to be natively baked
                if extract_vae or extract_audio_vae:
                    out = comfy.sd.load_checkpoint_guess_config(unet_path, output_vae=extract_vae, output_clip=False, output_clipvision=False, model_options=model_options)
                    model = out[0]
                    if extract_vae:
                        vae = out[2]
                        logger.info("◎ VAE: Extracted natively from UNET")
                        info_lines.append("◎ VAE: Baked from UNET")
                        if caching: _cache.put(baked_vae_key, vae)
                    if extract_audio_vae:
                        # Dedicated instantiation required for AudioVAE architecture
                        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
                        from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
                        audio_vae = AudioVAE(sd, metadata)
                        logger.info("◎ AUDIO VAE: Extracted natively from UNET")
                        info_lines.append("◎ AUDIO VAE: Baked from UNET")
                        if caching: _cache.put(baked_audio_vae_key, audio_vae)
                else:
                    model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
                    
                elapsed = time.time() - t0
                logger.info(f"◎ UNET: {unet_name} [{weight_dtype}] ({elapsed:.1f}s)")
                info_lines.append(f"◎ UNET: {unet_name} [{weight_dtype}] ({elapsed:.1f}s)")
                if caching:
                    _cache.put(unet_key, model)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load UNET '{unet_name}': {e}")
                
        # ════════════════════════════════════════════════════════════════
        # LOAD STANDALONE VIDEO VAE (ALBABIT-FIX)
        # ════════════════════════════════════════════════════════════════
        if not extract_vae:
            t0 = time.time()
            vae_path = _ensure_model_exists(vae_name, "vae", auto_download)
            if not vae_path:
                raise FileNotFoundError(f"❌ VAE not found: '{vae_name}'. Enable auto_download or install it manually.")

            vae_fp  = _file_fingerprint(vae_path)
            vae_key = f"vae:{vae_path}:{vae_fp}"

            if caching and _cache.has(vae_key):
                vae = _cache.get(vae_key)
                logger.info(f"◎ VAE from cache: {vae_name}")
                info_lines.append(f"◎ VAE: {vae_name} (cached)")
            else:
                try:
                    # ALBABIT-FIX: Using load_torch_file with return_metadata=True. 
                    # LTX 2.3 VAEs require metadata to correctly identify their 256-channel architecture.
                    # Without it, ComfyUI falls back to the 128-channel LTX 1.0 config, causing a size mismatch.
                    sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
                    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
                    
                    elapsed = time.time() - t0
                    logger.info(f"◎ VAE: {vae_name} ({elapsed:.1f}s)")
                    info_lines.append(f"◎ VAE: {vae_name} ({elapsed:.1f}s)")
                    if caching:
                        _cache.put(vae_key, vae)
                except Exception as e:
                    logger.error(f"❌ Failed to load VAE '{vae_name}': {e}")
                    raise RuntimeError(f"❌ Failed to load VAE '{vae_name}': {e}")

        # ════════════════════════════════════════════════════════════════
        # LOAD STANDALONE AUDIO VAE (ALBABIT-FIX)
        # ════════════════════════════════════════════════════════════════
        if not extract_audio_vae and audio_vae_name != "None":
            t0 = time.time()
            audio_vae_path = folder_paths.get_full_path("checkpoints", audio_vae_name)
            if not audio_vae_path: 
                audio_vae_path = folder_paths.get_full_path("vae", audio_vae_name)
            
            if audio_vae_path:
                audio_vae_fp = _file_fingerprint(audio_vae_path)
                audio_vae_key = f"audio_vae:{audio_vae_path}:{audio_vae_fp}"
                
                if caching and _cache.has(audio_vae_key):
                    audio_vae = _cache.get(audio_vae_key)
                    logger.info("◎ AUDIO VAE from cache")
                    info_lines.append(f"◎ AUDIO VAE: {audio_vae_name} (cached)")
                else:
                    try:
                        sd, metadata = comfy.utils.load_torch_file(audio_vae_path, return_metadata=True)
                        from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
                        audio_vae = AudioVAE(sd, metadata)
                        elapsed = time.time() - t0
                        logger.info(f"◎ AUDIO VAE: {audio_vae_name} ({elapsed:.1f}s)")
                        info_lines.append(f"◎ AUDIO VAE: {audio_vae_name} ({elapsed:.1f}s)")
                        if caching: _cache.put(audio_vae_key, audio_vae)
                    except Exception as e:
                        logger.error(f"❌ Failed to load standalone Audio VAE: {e}")
                        info_lines.append(f"❌ AUDIO VAE: Load failed ({e})")
            else:
                logger.warning(f"❌ Audio VAE file not found: {audio_vae_name}")
                info_lines.append(f"❌ AUDIO VAE: Not found")

        # ════════════════════════════════════════════════════════════════
        # 5. LOAD CLIP  (named slots → ordered paths → mtime cache key)
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        
        # Ensure all selected CLIPs exist/downloaded
        for slot, val in [("clip_l", clip_l), ("clip_g", clip_g), 
                          ("t5xxl", t5xxl), ("llm_encoder", llm_encoder), 
                          ("text_projection", text_projection)]:
             _ensure_model_exists(val, "text_encoders", auto_download)

        clip_paths = _assemble_clip_paths(resolved_type, clip_l, clip_g, t5xxl, llm_encoder, text_projection)

        # Allow missing paths ONLY IF we are injecting the UNET natively as a dualclip projection source
        is_gemma_ltx = (resolved_type in ("ltx", "ltxav") and llm_encoder and llm_encoder != "None" and "gemma" in llm_encoder.lower())

        if not clip_paths and not is_gemma_ltx:
            raise ValueError(
                f"❌ No CLIP encoders provided for architecture '{resolved_type}'. "
                f"Fill the required slot(s): "
                f"{', '.join(CLIP_SLOT_ORDER.get(resolved_type, ['clip_l']))}"
            )

        clip_fps     = ":".join(_file_fingerprint(p) for p in clip_paths)
        
        # ALBABIT-FIX: Native Gemma 3 DualCLIP Cache Identification
        if is_gemma_ltx:
            clip_key = f"clip:gemma:{llm_encoder}:{text_projection}:{unet_name}:{clip_fps}"
            clip_slot_used = ["llm_encoder (Gemma 3)"]
            if text_projection and text_projection != "None":
                clip_slot_used.append("text_projection")
        else:
            clip_key = f"clip:{':'.join(clip_paths)}:{resolved_type}:{clip_dtype}:{clip_fps}"
            clip_slot_used = []
            for slot, val in [("clip_l", clip_l), ("clip_g", clip_g),
                              ("t5xxl", t5xxl), ("llm_encoder", llm_encoder), 
                              ("text_projection", text_projection)]:
                if val and val not in ("None", ""):
                    clip_slot_used.append(slot)

        if caching and _cache.has(clip_key):
            clip = _cache.get(clip_key)
            logger.info(f"◎ CLIP from cache: {clip_slot_used}")
            info_lines.append(f"◎ CLIP: {'+'.join(clip_slot_used)} (cached)")
        else:
            
            # ALBABIT-FIX: Emulate native ComfyUI DualCLIP load for LTX 2.3 Gemma 3
            if is_gemma_ltx:
                clip_type_enum = getattr(comfy.sd.CLIPType, "LTXV", comfy.sd.CLIPType.STABLE_DIFFUSION)
                llm_path = folder_paths.get_full_path("text_encoders", llm_encoder)
                if not llm_path: raise FileNotFoundError(f"Missing Gemma model: {llm_encoder}")
                
                clip_paths = [llm_path]
                if text_projection and text_projection != "None":
                    proj_path = folder_paths.get_full_path("text_encoders", text_projection)
                    if not proj_path: raise FileNotFoundError(f"Missing projection model: {text_projection}")
                    clip_paths.append(proj_path)
                    logger.info("◎ ALBABIT-FIX: Using standalone text_projection safetensors file.")
                else:
                    clip_paths.append(unet_path)
                    logger.info("◎ ALBABIT-FIX: Extracting text_projection natively from UNET.")
            else:
                clip_type_enum = get_clip_type_enum(resolved_type)
                
            clip_model_opts = {}
            if clip_load_device:
                clip_model_opts["load_device"] = clip_load_device
            dtype_map = {
                "fp16": torch.float16, "bf16": torch.bfloat16,
                "fp8_e4m3fn": torch.float8_e4m3fn, "fp32": torch.float32,
            }
            if clip_dtype in dtype_map:
                clip_model_opts["dtype"] = dtype_map[clip_dtype]

            try:
                clip = comfy.sd.load_clip(
                    ckpt_paths=clip_paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=clip_type_enum,
                    model_options=clip_model_opts if clip_model_opts else {},
                )
                elapsed = time.time() - t0
                logger.info(
                    f"◎ CLIP: {'+'.join(clip_slot_used)} "
                    f"[type={resolved_type}, dtype={clip_dtype}] ({elapsed:.1f}s)"
                )
                info_lines.append(
                    f"◎ CLIP: {'+'.join(clip_slot_used)} [{clip_dtype}] ({elapsed:.1f}s)"
                )
                if caching:
                    _cache.put(clip_key, clip)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load CLIP: {e}")

        # ════════════════════════════════════════════════════════════════
        # 6. LOAD LATENT UPSCALE MODEL (ALBABIT-FIX)
        # ════════════════════════════════════════════════════════════════
        upscale_model = None
        if upscale_model_name and upscale_model_name != "None":
            t0 = time.time()
            upscale_path = folder_paths.get_full_path("latent_upscale_models", upscale_model_name)
            if not upscale_path:
                logger.warning(f"❌ Latent Upscale Model not found: '{upscale_model_name}'")
                info_lines.append(f"❌ Upscale Model: Not found")
            else:
                upscale_fp = _file_fingerprint(upscale_path)
                upscale_key = f"upscale_model:{upscale_path}:{upscale_fp}"

                if caching and _cache.has(upscale_key):
                    upscale_model = _cache.get(upscale_key)
                    logger.info(f"◎ Latent Upscale Model from cache: {upscale_model_name}")
                    info_lines.append(f"◎ Upscale Model: {upscale_model_name} (cached)")
                else:
                    try:
                        sd, metadata = comfy.utils.load_torch_file(upscale_path, safe_load=True, return_metadata=True)
                        
                        # ALBABIT-FIX: Native Hunyuan/LTX upscale model routing 
                        # Imported locally to prevent crashes on older ComfyUI versions lacking these modules
                        if "blocks.0.block.0.conv.weight" in sd:
                            from comfy.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel
                            config = {
                                "in_channels": sd["in_conv.conv.weight"].shape[1],
                                "out_channels": sd["out_conv.conv.weight"].shape[0],
                                "hidden_channels": sd["in_conv.conv.weight"].shape[0],
                                "num_blocks": len([k for k in sd.keys() if k.startswith("blocks.") and k.endswith(".block.0.conv.weight")]),
                                "global_residual": False,
                            }
                            upscale_model = HunyuanVideo15SRModel("720p", config)
                            upscale_model.load_sd(sd)
                        elif "up.0.block.0.conv1.conv.weight" in sd:
                            from comfy.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel
                            sd = {key.replace("nin_shortcut", "nin_shortcut.conv", 1): value for key, value in sd.items()}
                            config = {
                                "z_channels": sd["conv_in.conv.weight"].shape[1],
                                "out_channels": sd["conv_out.conv.weight"].shape[0],
                                "block_out_channels": tuple(sd[f"up.{i}.block.0.conv1.conv.weight"].shape[0] for i in range(len([k for k in sd.keys() if k.startswith("up.") and k.endswith(".block.0.conv1.conv.weight")]))),
                            }
                            upscale_model = HunyuanVideo15SRModel("1080p", config)
                            upscale_model.load_sd(sd)
                        elif "post_upsample_res_blocks.0.conv2.bias" in sd:
                            from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
                            config = json.loads(metadata["config"])
                            upscale_model = LatentUpsampler.from_config(config).to(dtype=comfy.model_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32]))
                            upscale_model.load_state_dict(sd)
                        else:
                            logger.warning(f"❌ Unrecognized upscale model architecture for: '{upscale_model_name}'")

                        if upscale_model is not None:
                            elapsed = time.time() - t0
                            logger.info(f"◎ Latent Upscale Model: {upscale_model_name} ({elapsed:.1f}s)")
                            info_lines.append(f"◎ Upscale Model: {upscale_model_name} ({elapsed:.1f}s)")
                            if caching: 
                                _cache.put(upscale_key, upscale_model)

                    except Exception as e:
                        logger.error(f"❌ Failed to load Latent Upscale Model '{upscale_model_name}': {e}")
                        info_lines.append(f"❌ Upscale Model: Load failed ({e})")

        # ════════════════════════════════════════════════════════════════
        # 7. APPLY LoRA STACK
        #    Priority: upstream lora_stack → built-in lora_1/2/3
        # ════════════════════════════════════════════════════════════════
        combined_loras = list(lora_stack) if lora_stack else []
        for name, ms, cs in [
            (lora_1, lora_1_model_str, lora_1_clip_str),
            (lora_2, lora_2_model_str, lora_2_clip_str),
            (lora_3, lora_3_model_str, lora_3_clip_str),
        ]:
            if name and name != "None":
                combined_loras.append((name, float(ms), float(cs)))

        applied_loras = []
        for i, (lora_name, model_str, clip_str) in enumerate(combined_loras, 1):
            if model_str == 0 and clip_str == 0:
                continue
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                msg = f"◎ LoRA not found: '{lora_name}'"
                if lora_on_error == "raise":
                    raise FileNotFoundError(msg)
                logger.warning(f"◎ {msg} — Skipping.")
                info_lines.append(f"◎ LoRA {i}: {lora_name} (not found)")
                continue
            try:
                t0 = time.time()
                lora_data = comfy.utils.load_torch_file(lora_path)
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip, lora_data, model_str, clip_str
                )
                elapsed = time.time() - t0
                logger.info(
                    f"◎ LoRA {i}: {lora_name} (model={model_str}, clip={clip_str}, {elapsed:.1f}s)"
                )
                info_lines.append(
                    f"◎ LoRA {i}: {lora_name} [m={model_str} c={clip_str}] ({elapsed:.1f}s)"
                )
                applied_loras.append({"name": lora_name, "model_str": model_str,
                                      "clip_str": clip_str})
            except Exception as e:
                msg = f"Failed to apply LoRA '{lora_name}': {e}"
                if lora_on_error == "raise":
                    raise RuntimeError(f"❌ {msg}")
                logger.warning(f"◎ {msg} — Skipping.")
                info_lines.append(f"◎ LoRA {i}: {lora_name} (failed)")

        # ════════════════════════════════════════════════════════════════
        # 8. LOAD CONTROLNET
        # ════════════════════════════════════════════════════════════════
        controlnet = None
        if controlnet_name and controlnet_name != "None":
            cn_path = folder_paths.get_full_path("controlnet", controlnet_name)
            if not cn_path:
                logger.warning(f"◎ ControlNet not found: '{controlnet_name}'")
                info_lines.append(f"◎ ControlNet: {controlnet_name} (not found)")
            else:
                try:
                    t0 = time.time()
                    controlnet = comfy.sd.load_controlnet(cn_path)
                    elapsed = time.time() - t0
                    logger.info(
                        f"◎ ControlNet: {controlnet_name} "
                        f"(str={controlnet_strength} range={controlnet_start}-{controlnet_end}, "
                        f"{elapsed:.1f}s)"
                    )
                    info_lines.append(
                        f"◎ ControlNet: {controlnet_name} "
                        f"[str={controlnet_strength} {controlnet_start}-{controlnet_end}] "
                        f"({elapsed:.1f}s)"
                    )
                    try:
                        controlnet.radiance_strength = float(controlnet_strength)
                        controlnet.radiance_start    = float(controlnet_start)
                        controlnet.radiance_end      = float(controlnet_end)
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"◎ ControlNet load failed '{controlnet_name}': {e}")
                    info_lines.append(f"◎ ControlNet: {controlnet_name} (failed)")

        # ════════════════════════════════════════════════════════════════
        # 9. BUILD OUTPUTS
        # ════════════════════════════════════════════════════════════════
        total_ms = round((time.time() - load_start) * 1000)
        summary  = (f"◎ Load complete in {total_ms / 1000:.1f}s"
                    + (f" (cache: {_cache.size})" if caching else ""))
        logger.info(summary)
        info_lines.append(summary)

        load_info = "\n".join(info_lines)

        # model_meta — structured JSON for downstream QC / analytics nodes
        model_meta = {
            "arch":          resolved_type,
            "detected":      detected_type is not None,
            "unet_file":     unet_name,
            "weight_dtype":  weight_dtype,
            "clip_slots":    clip_slot_used,
            "clip_dtype":    clip_dtype,
            "offload_mode":  offload_mode,
            "latent_ch":     LATENT_CHANNELS.get(resolved_type, 4),
            "latent_format": latent_fmt,
            "vram_est_gb":   est,
            "loras":         applied_loras,
            "controlnet":    controlnet_name if controlnet else None,
            "upscale_model": upscale_model_name if upscale_model else None, # ALBABIT-FIX
            "load_ms":       total_ms,
            "cached_unet":   unet_cache_hit,  # FIX 4: True only when loaded from cache
        }

        # Output the accumulated lora list (for chaining downstream loaders
        # or QC nodes that want to know what was applied)
        out_lora_stack = [(e["name"], e["model_str"], e["clip_str"])
                          for e in applied_loras] if applied_loras else None

        # ALBABIT-FIX: Return audio_vae and upscale_model in the stack
        return (
            model,
            clip,
            vae,
            audio_vae, 
            controlnet,
            out_lora_stack,
            upscale_model, 
            load_info,
            latent_fmt,
            json.dumps(model_meta, indent=2),
        )


# ═══════════════════════════════════════════════════════════════════════════════
class RadianceControlNetApply:
    """
    Advanced ControlNet application node for the Radiance suite.
    - Gracefully bypasses if CONTROL_NET is None (prevents AttributeError crashes).
    - Supports standard start/end percent and strength controls.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Global strength of the control effect."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Percentage of the generation where control starts (0.0 = beginning)."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Percentage of the generation where control ends (1.0 = end)."}),
                "control_type": (["auto"] + list(UNION_CONTROLNET_TYPES.keys()), {"default": "auto",
                    "tooltip": "For Union ControlNets (like Flux), select the specific control mode (Canny, Depth, etc.)."}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)  # FIX 7: was missing — ComfyUI showed generic labels
    FUNCTION = "apply_controlnet"
    CATEGORY = "FXTD Studios/Radiance/Generate"

    def apply_controlnet(self, conditioning, control_net, image, strength, start_percent, end_percent, control_type="auto"):
        # 1. Graceful Bypass: If no control_net or zero strength, just return the input conditioning.
        if control_net is None:
            logger.info("◎ Radiance Control: No ControlNet connected. Bypassing.")
            return (conditioning, )
            
        if strength == 0:
            return (conditioning, )

        # 2. Set Union ControlNet Type (if applicable)
        control_net = control_net.copy()
        type_number = UNION_CONTROLNET_TYPES.get(control_type, -1)
        if type_number >= 0:
            control_net.set_extra_arg("control_type", [type_number])
        else:
            control_net.set_extra_arg("control_type", [])

        # 3. Advanced Application (Standard ComfyUI Logic with added safety)
        c = []
        try:
            for t in conditioning:
                n = [t[0], t[1].copy()]
                c_net = control_net.copy().set_cond_hint(image, strength, (start_percent, end_percent))
                if 'control' in n[1]:
                    c_net.set_previous_controlnet(n[1]['control'])
                n[1]['control'] = c_net
                n[1]['control_apply_strength'] = strength
                c.append(n)
            return (c, )
        except Exception as e:
            logger.error(f"❌ Radiance Control: Failed to apply ControlNet: {e}")
            return (conditioning, )


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

# FIX 1: NODE_CLASS_MAPPINGS keys must be plain ASCII identifiers.
# The ◎ prefix belongs only in NODE_DISPLAY_NAME_MAPPINGS (the user-visible label).
# Having it in the type key breaks ComfyUI workflow JSON serialization and node lookup.
NODE_CLASS_MAPPINGS = {
    "RadianceUnifiedLoader": RadianceUnifiedLoader,
    "RadianceLoraStack":     RadianceLoraStack,
    "RadianceControlApply":  RadianceControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceUnifiedLoader": "◎ Radiance Unified Loader",
    "RadianceLoraStack":     "◎ Radiance LoRA Stack",
    "RadianceControlApply":  "◎ Radiance Control",
}