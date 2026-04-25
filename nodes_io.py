import os
import gc
import glob
import json
import time
import logging
import datetime
import subprocess  # nosec B404
import tempfile
from typing import Dict, Any, Optional, List, Tuple

# Enable cv2's EXR codec before first use. The codec is initialised lazily
# (on first EXR read/write attempt, not at import time), so this is effective
# even if cv2 was already imported by ComfyUI. Without this flag, cv2 prints
# a per-frame "OpenEXR codec is disabled" warning to stderr whenever
# write_exr_robust() triggers cv2's codec scan as part of its internal flow.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import torch
import numpy as np
import cv2

from . import color_utils
try:
    from .path_utils import safe_join, get_safe_output_dir, get_safe_input_path, get_next_index
except ImportError:
    from path_utils import safe_join, get_safe_output_dir, get_safe_input_path, get_next_index

import folder_paths

try:
    from .hdr.io import write_exr_robust, write_hdr_rgbe
except ImportError:
    try:
        from hdr.io import write_exr_robust, write_hdr_rgbe
    except ImportError:
        write_exr_robust = None
        write_hdr_rgbe = None

# write_exr_multipart is a newer addition — import separately so a missing
# symbol does not poison the write_exr_robust / write_hdr_rgbe imports above.
try:
    from .hdr.io import write_exr_multipart
except ImportError:
    try:
        from hdr.io import write_exr_multipart
    except ImportError:
        write_exr_multipart = None

# ALBABIT-FIX: PIL used for 8-bit PNG tEXt metadata chunks and RGBA alpha writing.
try:
    from PIL import Image as PILImage
    from PIL.PngImagePlugin import PngInfo as PNGInfo
    _HAS_PIL = True
except ImportError:
    PILImage = None
    PNGInfo  = None
    _HAS_PIL = False


logger = logging.getLogger("Radiance.io")

# ALBABIT-FIX: Strip surrounding quotes from path strings.
# Windows "Copy as path" (Shift+Right-click) wraps paths in double-quotes.
def _strip_path_quotes(path: str) -> str:
    return path.strip().strip('"').strip("'")

# ── Memory budget helpers ──────────────────────────────────────────────────────

def _available_ram_bytes() -> int:
    """Return available system RAM in bytes. Falls back to 4 GB if psutil absent."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return 4 * 1024 ** 3


def _check_memory_budget(n_frames: int, h: int, w: int, c: int,
                          dtype_bytes: int = 4, label: str = "") -> None:
    """
    Warn / raise before allocating a large tensor batch.
    Peak during torch.stack = 2× the final tensor size.
    Raises MemoryError if allocation would exceed 90% of available RAM.
    Logs a WARNING if it would exceed 60%.
    """
    final_bytes = n_frames * h * w * c * dtype_bytes
    peak_bytes  = final_bytes * 2
    available   = _available_ram_bytes()
    final_gb    = final_bytes / 1024 ** 3
    peak_gb     = peak_bytes  / 1024 ** 3
    avail_gb    = available   / 1024 ** 3

    tag = f"[Cinema Read{' ' + label if label else ''}]"

    if peak_bytes > available * 0.90:
        raise MemoryError(
            f"{tag} Loading {n_frames} frames at {w}×{h} would require "
            f"~{peak_gb:.1f} GB RAM (2× peak for tensor construction), "
            f"but only {avail_gb:.1f} GB available. "
            f"Reduce frame_limit or use a smaller resolution. "
            f"Final tensor would be {final_gb:.1f} GB."
        )
    if peak_bytes > available * 0.60:
        logger.warning(
            f"{tag} Loading {n_frames}×{w}×{h} will use "
            f"~{peak_gb:.1f} GB ({peak_gb/avail_gb*100:.0f}% of available RAM). "
            f"Consider setting frame_limit to reduce memory usage."
        )


class RadianceType(str):
    def __ne__(self, __value: object) -> bool:
        return False

image_video_type = RadianceType("IMAGE,VIDEO")

# ═══════════════════════════════════════════════════════════════════════════════
#                         COLOR SPACE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_COLORSPACES = [
    "Linear (sRGB)",
    "sRGB (Standard)",
    "ARRI LogC3",
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "DaVinci Intermediate",
    "ACEScg",
    "ACEScct",
]


def apply_input_transform(img_tensor: torch.Tensor, colorspace: str) -> torch.Tensor:
    if colorspace == "Linear (sRGB)" or colorspace == "ACEScg": return img_tensor
    elif colorspace == "sRGB (Standard)": return color_utils.tensor_srgb_to_linear(img_tensor)

    device = img_tensor.device
    img_np = img_tensor.cpu().numpy()
    transform_map = {
        "ARRI LogC3": color_utils.logc3_to_linear,
        "ARRI LogC4": color_utils.logc4_to_linear,
        "Sony S-Log3": color_utils.slog3_to_linear,
        "Panasonic V-Log": color_utils.vlog_to_linear,
        "DaVinci Intermediate": color_utils.davinci_intermediate_to_linear,
        "ACEScct": color_utils.acescct_to_linear,
    }
    fn = transform_map.get(colorspace)
    out_np = fn(img_np) if fn else img_np
    return torch.from_numpy(out_np).to(device)


def apply_output_transform(img_tensor: torch.Tensor, colorspace: str, broadcast_safe: bool = True) -> torch.Tensor:
    is_display_space = colorspace in ["sRGB (Standard)"] or ("sRGB" in colorspace and "Linear" not in colorspace)
    # BUG 1 FIX: broadcast_safe ACES tonemap must ONLY run for display-referred
    # colorspaces (sRGB). For Linear (sRGB) and ACEScg the data is scene-referred —
    # the tonemap would bake display values into the float tensor, silently corrupting
    # any linear HDR EXR export. Gate on is_display_space explicitly.
    if broadcast_safe and is_display_space:
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        x = img_tensor
        img_tensor = torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

    if colorspace == "Linear (sRGB)" or colorspace == "ACEScg":
        # HDR FIX: Never hard-clamp linear data even if broadcast_safe is on.
        # Scene-linear highlights legitimately exceed 1.0 and must be preserved for EXR.
        return img_tensor
    elif colorspace == "sRGB (Standard)":
        return color_utils.tensor_linear_to_srgb(img_tensor)

    device = img_tensor.device
    img_np = img_tensor.cpu().numpy()
    transform_map = {
        "ARRI LogC3": color_utils.linear_to_logc3,
        "ARRI LogC4": color_utils.linear_to_logc4,
        "Sony S-Log3": color_utils.linear_to_slog3,
        "Panasonic V-Log": color_utils.linear_to_vlog,
        "DaVinci Intermediate": color_utils.linear_to_davinci_intermediate,
        "ACEScct": color_utils.linear_to_acescct,
    }
    fn = transform_map.get(colorspace)
    out_np = fn(img_np) if fn else img_np
    return torch.from_numpy(out_np).to(device)

# ═══════════════════════════════════════════════════════════════════════════════
#                         HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_audio_ffmpeg(video_path: str) -> dict | None:
    # BUG 2 FIX: previous implementation had os.unlink(tmp_path) inside the try block
    # only on the success path. Exceptions (ffmpeg fail, parse error, etc.) after the
    # tmp file was created left orphaned .wav files in the system temp dir.
    # Now the tmp is cleaned up in finally regardless of exit path.
    try:
        probe_cmd = ["ffprobe", "-v", "quiet", "-select_streams", "a:0", "-show_entries", "stream=codec_type,sample_rate,channels", "-of", "json", video_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0: return None
        probe_data = json.loads(result.stdout)
        streams = probe_data.get("streams", [])
        if not streams: return None
        sample_rate = int(streams[0].get("sample_rate", 44100))
        channels = int(streams[0].get("channels", 2))
    except Exception as e:
        logger.debug(f"[Cinema Read] ffprobe failed for {video_path}: {e}")
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        extract_cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_f32le", "-ar", str(sample_rate), "-ac", str(channels), tmp_path]
        subprocess.run(extract_cmd, capture_output=True, timeout=120)
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 100:
            return None
        import struct
        with open(tmp_path, "rb") as f:
            data = f.read()
        data_offset = data.find(b"data")
        if data_offset == -1:
            return None
        data_offset += 4
        data_size = struct.unpack_from("<I", data, data_offset)[0]
        data_offset += 4
        raw_audio = data[data_offset : data_offset + data_size]
        n_samples = len(raw_audio) // 4
        samples = np.frombuffer(raw_audio, dtype=np.float32)
        samples_per_channel = n_samples // channels
        samples = samples[: samples_per_channel * channels]
        waveform = samples.reshape(samples_per_channel, channels).T
        waveform_tensor = torch.from_numpy(waveform.copy()).unsqueeze(0)
        return {"waveform": waveform_tensor, "sample_rate": sample_rate}
    except Exception as e:
        logger.debug(f"[Cinema Read] Audio extraction failed for {video_path}: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

def find_video_path(data: Any) -> Optional[str]:
    if isinstance(data, str):
        if data.lower().endswith((".mp4", ".mov", ".gif", ".webp", ".avi", ".mkv", ".webm")): return data
        return None
    if isinstance(data, dict):
        for key in ["source", "filename", "video", "gif", "full_path"]:
            if key in data:
                res = find_video_path(data[key])
                if res: return res
        for val in data.values():
            res = find_video_path(val)
            if res: return res
    if isinstance(data, list):
        for item in data:
            res = find_video_path(item)
            if res: return res
    return None


# Known video extensions — fast-path hint; _is_video_file() handles unknowns via ffprobe.
_VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf", ".m4v", ".ts",
    ".mts", ".m2ts", ".mpg", ".mpeg", ".wmv", ".flv", ".f4v",
    ".3gp", ".gif", ".webp", ".r3d", ".braw", ".arri",
}

# Known image/sequence extensions — used to fast-skip video probe.
_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tiff", ".tif", ".dpx",
    ".cin", ".bmp", ".psd", ".svg", ".heic", ".avif",
}


def _is_video_file(path: str) -> bool:
    """
    Universal video detection — does NOT rely on extension whitelist.
    Strategy (fastest → most reliable):
      1. Known IMAGE extension → False immediately
      2. Known VIDEO extension → True immediately
      3. Unknown extension → ffprobe stream detection
      4. ffprobe unavailable → cv2.VideoCapture fallback
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in _IMAGE_EXTENSIONS:
        return False
    if ext in _VIDEO_EXTENSIONS:
        return True

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type,nb_frames,r_frame_rate",
             "-of", "json", path],
            capture_output=True, text=True, timeout=8,
        )
        if probe.returncode == 0:
            info = json.loads(probe.stdout)
            streams = info.get("streams", [])
            if streams and streams[0].get("codec_type") == "video":
                nb = streams[0].get("nb_frames", "0")
                try:
                    return int(nb) > 1
                except (ValueError, TypeError):
                    return True
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        pass

    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps > 0 and total > 1
    except Exception:
        return False

def _load_video_frames(video_path: str, input_colorspace: str = "sRGB (Standard)") -> torch.Tensor | None:
    if not os.path.exists(video_path): return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames: return None
    frames_np = np.stack(frames).astype(np.float32) / 255.0
    return apply_input_transform(torch.from_numpy(frames_np), input_colorspace)

# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE: DIGITAL CINEMA READ (UNIVERSAL)
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceDigitalCinemaRead:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {"default": "C:/Footage/shot_01.mp4",
                    "tooltip": (
                        "Path to a video file (any extension), image sequence folder, "
                        "or single image. Absolute paths are supported. "
                        "For sequences: point to the folder or a single file in the sequence.\n"
                        "Forward slashes and backslashes are both accepted. "
                        "Paths wrapped in quotes (e.g. \"D:\\Footage\\shot.mp4\") are also accepted."
                    ),
                }),
                "read_mode": (
                    ["Auto", "Video", "Sequence", "Single Frame"],
                    {
                        "default": "Auto",
                        "tooltip": (
                            "Auto  = detect from file type (default).\n"
                            "Video = force video decode for ANY extension "
                            "(.r3d, .braw, .mxf, .ts, etc.) — uses ffprobe+cv2.\n"
                            "Sequence = force image sequence even if path looks like video.\n"
                            "Single Frame = extract exactly ONE frame from a video or load "
                            "one image. Use frame_number to pick which frame."
                        ),
                    },
                ),
                "start_frame": ("INT", {"default": 1, "min": 0,
                    "tooltip": "First frame to read (1-based, 0 = from the very beginning). For Video/Sequence modes."}),
                "frame_limit": ("INT", {"default": 0, "min": 0,
                    "tooltip": "Maximum frames to load. 0 = load all."}),
                "frame_number": ("INT", {"default": 1, "min": 1, "max": 999999,
                    "tooltip": (
                        "Frame to extract in 'Single Frame' mode (1-based).\n"
                        "• Video source: seeks to the specified frame via VideoCapture.\n"
                        "• Image file (EXR, PNG, etc.): ignored — a single image has only one frame.\n"
                        "Has no effect in Auto, Video, or Sequence modes."
                    ),
                }),
                # ALBABIT-FIX: Default changed from "sRGB (Standard)" to "Linear (sRGB)".
                # Radiance nodes target linear HDR workflows (EXR sequences);
                # "Linear (sRGB)" is the correct default for the majority of use cases.
                "input_colorspace": (INPUT_COLORSPACES, {"default": "Linear (sRGB)"}),
                "fps_override": ("FLOAT", {"default": 0.0, "min": 0.0,
                    "tooltip": (
                        "Override detected FPS. 0 = auto-detect (video) or 24.0 (sequence).\n"
                        "Not available in Single Frame mode — FPS has no meaning for a still image."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "FLOAT", "AUDIO", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count", "width", "height", "fps", "audio", "video")
    FUNCTION = "read"
    CATEGORY = "FXTD Studios/Radiance/IO"

    def _load_single_exr(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Robust EXR loader — memory-optimised: channels written in-place, no intermediate stack copy."""
        img, alpha_img, depth_img = None, None, None
        try:
            import OpenEXR, Imath
            exr_file = OpenEXR.InputFile(file_path)
            header   = exr_file.header()
            dw       = header["dataWindow"]
            width, height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
            channels = list(header["channels"].keys())
            pt       = Imath.PixelType(Imath.PixelType.FLOAT)

            if "R" in channels and "G" in channels and "B" in channels:
                # Pre-allocate output array, fill channels in-place, del each buffer
                # immediately to avoid holding r+g+b+img in memory simultaneously.
                img = np.empty((height, width, 3), dtype=np.float32)
                r   = np.frombuffer(exr_file.channel("R", pt), dtype=np.float32).copy().reshape(height, width)
                img[..., 0] = r;  del r
                g   = np.frombuffer(exr_file.channel("G", pt), dtype=np.float32).copy().reshape(height, width)
                img[..., 1] = g;  del g
                b   = np.frombuffer(exr_file.channel("B", pt), dtype=np.float32).copy().reshape(height, width)
                img[..., 2] = b;  del b

                if "A" in channels:
                    alpha_img = np.frombuffer(exr_file.channel("A", pt), dtype=np.float32).copy().reshape(height, width)
                else:
                    alpha_img = np.ones((height, width), dtype=np.float32)

            elif "Y" in channels:
                y   = np.frombuffer(exr_file.channel("Y", pt), dtype=np.float32).copy().reshape(height, width)
                img = np.empty((height, width, 3), dtype=np.float32)
                img[..., 0] = y; img[..., 1] = y; img[..., 2] = y;  del y
                if "A" in channels:
                    alpha_img = np.frombuffer(exr_file.channel("A", pt), dtype=np.float32).copy().reshape(height, width)
                else:
                    alpha_img = np.ones((height, width), dtype=np.float32)

            for dname in ["Z", "Depth", "depth", "z"]:
                if dname in channels:
                    depth_img = np.frombuffer(exr_file.channel(dname, pt), dtype=np.float32).copy().reshape(height, width)
                    break
            exr_file.close()
            return img, alpha_img, depth_img

        except (ImportError, Exception) as e:
            logger.debug(f"[Cinema Read] OpenEXR backend failed for {file_path}: {e}")

        try:
            import imageio.v3 as iio
            raw = iio.imread(file_path)
            return np.asarray(raw, dtype=np.float32), None, None
        except Exception as e:
            logger.debug(f"[Cinema Read] imageio backend failed for {file_path}: {e}")
        return None, None, None

    def read(self, source_path, read_mode="Auto", start_frame=1, frame_limit=0,
             frame_number=1, input_colorspace="sRGB (Standard)", fps_override=0.0):

        source_path = _strip_path_quotes(source_path)

        # ── Path resolution ────────────────────────────────────────────────────
        resolved = None
        for dir_fn in [folder_paths.get_input_directory, folder_paths.get_output_directory]:
            try:
                candidate = get_safe_input_path(dir_fn(), source_path, allow_absolute=True)
                if os.path.exists(candidate):
                    resolved = candidate
                    break
            except Exception:
                continue
        source_path = resolved or source_path

        # ── Mode detection ─────────────────────────────────────────────────────
        if read_mode == "Auto":
            is_video    = _is_video_file(source_path)
            is_sequence = not is_video
        elif read_mode == "Video":
            is_video    = True
            is_sequence = False
        elif read_mode == "Single Frame":
            is_video    = False
            is_sequence = False
        else:  # "Sequence"
            is_video    = False
            is_sequence = True

        # ══════════════════════════════════════════════════════════════════════
        # SINGLE FRAME PATH (image file — EXR, HDR, PNG, JPG, etc.)
        # ══════════════════════════════════════════════════════════════════════
        if read_mode == "Single Frame":
            ext = os.path.splitext(source_path)[1].lower()
            img     = None
            mask_np = None

            # ALBABIT-FIX: Video source in Single Frame mode — seek to frame_number via VideoCapture.
            # cv2.imread() cannot decode video containers; VideoCapture is required for seeking.
            if _is_video_file(source_path):
                cap = cv2.VideoCapture(source_path)
                if not cap.isOpened():
                    raise IOError(
                        f"[Cinema Read] Cannot open video '{source_path}'. "
                        f"Exotic formats (.r3d, .braw, .mxf) require ffmpeg in PATH."
                    )
                total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                detected_fps = cap.get(cv2.CAP_PROP_FPS)
                if detected_fps < 1.0 or detected_fps > 1000.0:
                    detected_fps = 24.0
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                seek   = max(0, min(frame_number - 1, max(0, total - 1)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise IOError(
                        f"[Cinema Read] Could not read frame {frame_number} from '{source_path}' "
                        f"(total: {total} frames)."
                    )
                img_f32 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frame_t = torch.from_numpy(img_f32[:, :, :3].copy()).unsqueeze(0);  del img_f32
                images  = apply_input_transform(frame_t, input_colorspace)
                mask    = torch.ones((1, height, width), dtype=torch.float32)
                logger.info(f"[Cinema Read] Single Frame #{frame_number}/{total} from video '{os.path.basename(source_path)}' ({width}×{height})")
                return (images, mask, 1, width, height, float(detected_fps), None, source_path)

            if ext in (".exr", ".hdr"):
                try:
                    rgb_np, alpha_np, _ = self._load_single_exr(source_path)
                    if rgb_np is None:
                        raise IOError(
                            f"[Cinema Read] Cannot load EXR '{source_path}'. "
                            f"Install OpenEXR (pip install OpenEXR) or imageio for EXR support."
                        )
                    img     = rgb_np
                    mask_np = alpha_np if alpha_np is not None else np.ones(img.shape[:2], dtype=np.float32)
                except IOError:
                    raise
                except Exception as e:
                    raise IOError(f"[Cinema Read] EXR/HDR load failed for '{source_path}': {e}") from e
            else:
                raw = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    raise IOError(f"[Cinema Read] Cannot open '{source_path}'.")
                if raw.ndim == 2:
                    img     = np.stack([raw, raw, raw], axis=-1);  del raw
                    mask_np = np.ones(img.shape[:2], dtype=np.float32)
                elif raw.shape[-1] == 4:
                    mask_np = raw[..., 3].astype(np.float32) / (255.0 if raw.dtype == np.uint8 else 65535.0)
                    img     = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB);  del raw
                else:
                    img     = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB);  del raw
                    mask_np = np.ones(img.shape[:2], dtype=np.float32)

            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            else:
                img = img.astype(np.float32)

            height, width = img.shape[:2]
            # ALBABIT-FIX: fps_override is ignored in Single Frame mode — FPS has no meaning for a still image.
            fps = 24.0
            frame_t = torch.from_numpy(img[:, :, :3].copy()).unsqueeze(0)
            del img
            images = apply_input_transform(frame_t, input_colorspace)
            mask   = torch.from_numpy(mask_np).unsqueeze(0)
            logger.info(f"[Cinema Read] Single Frame from '{os.path.basename(source_path)}' ({width}×{height})")
            return (images, mask, 1, width, height, float(fps), None, source_path)

        # ══════════════════════════════════════════════════════════════════════
        # VIDEO PATH
        # ══════════════════════════════════════════════════════════════════════
        if is_video:
            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                raise IOError(
                    f"[Cinema Read] Cannot open '{source_path}'. "
                    f"Exotic formats (.r3d, .braw, .mxf) require ffmpeg in PATH."
                )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            detected_fps = cap.get(cv2.CAP_PROP_FPS)
            if detected_fps < 1.0 or detected_fps > 1000.0:
                detected_fps = 24.0
            fps = fps_override if fps_override > 0 else detected_fps

            # ── Full video range ──────────────────────────────────────────────
            cv2_start      = max(0, start_frame - 1)
            frames_to_read = frame_limit if frame_limit > 0 else (total_frames - cv2_start)
            frames_to_read = max(0, min(frames_to_read, total_frames - cv2_start))

            _check_memory_budget(frames_to_read, height, width, 3, dtype_bytes=4, label="Video")

            cap.set(cv2.CAP_PROP_POS_FRAMES, cv2_start)
            frames_np = np.empty((frames_to_read, height, width, 3), dtype=np.uint8)
            actual    = 0
            for i in range(int(frames_to_read)):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_np[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                actual += 1
            cap.release()

            if actual == 0:
                raise ValueError(f"[Cinema Read] No frames decoded from '{source_path}'.")

            if actual < frames_to_read:
                frames_np = frames_np[:actual]

            float_np = frames_np.astype(np.float32) / 255.0
            del frames_np
            images   = apply_input_transform(torch.from_numpy(float_np), input_colorspace)
            del float_np
            mask     = torch.ones((actual, height, width), dtype=torch.float32)
            audio    = _extract_audio_ffmpeg(source_path)
            gc.collect()

            logger.info(f"[Cinema Read] Video: {actual} frames "
                        f"'{os.path.basename(source_path)}' ({width}×{height} @ {fps:.3f}fps)")
            return (images, mask, actual, width, height, float(fps), audio, source_path)

        # ══════════════════════════════════════════════════════════════════════
        # SEQUENCE / IMAGE PATH
        # ══════════════════════════════════════════════════════════════════════
        if os.path.isdir(source_path):
            folder, pattern = source_path, "*"
        else:
            folder, pattern = os.path.dirname(source_path), os.path.basename(source_path)

        files = sorted(glob.glob(os.path.join(folder, pattern)))
        files = [f for f in files if os.path.splitext(f)[1].lower() in _IMAGE_EXTENSIONS]

        start_idx = 0
        if start_frame > 1:
            num_str4 = str(start_frame).zfill(4)
            num_str  = str(start_frame)
            for idx, f in enumerate(files):
                bn = os.path.basename(f)
                if num_str4 in bn or num_str in bn:
                    start_idx = idx
                    break

        files = files[start_idx:]
        if frame_limit > 0:
            files = files[:frame_limit]

        if not files:
            raise ValueError(f"[Cinema Read] No image files found in: {source_path}")

        def _probe_dims(fpath):
            ext = os.path.splitext(fpath)[1].lower()
            if ext in (".exr", ".hdr"):
                rgb, _, _ = self._load_single_exr(fpath)
                if rgb is not None:
                    return rgb.shape[0], rgb.shape[1], rgb.shape[2]
                return None, None, None
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                c    = 3 if img.ndim == 2 else (3 if img.shape[-1] != 4 else 4)
                return h, w, c
            return None, None, None

        h0, w0, c0 = _probe_dims(files[0])
        if h0 is None:
            raise ValueError(f"[Cinema Read] Cannot read first file: {files[0]}")

        n_frames = len(files)
        c_out    = 3

        _check_memory_budget(n_frames, h0, w0, c_out, dtype_bytes=4, label="Sequence")

        batch_out  = torch.empty(n_frames, h0, w0, c_out, dtype=torch.float32)
        batch_mask = torch.empty(n_frames, h0, w0,         dtype=torch.float32)
        loaded     = 0

        for i, fpath in enumerate(files):
            ext     = os.path.splitext(fpath)[1].lower()
            img     = None
            mask_np = None

            if ext in (".exr", ".hdr"):
                try:
                    rgb_np, alpha_np, _ = self._load_single_exr(fpath)
                    if rgb_np is not None:
                        img     = rgb_np
                        mask_np = alpha_np if alpha_np is not None else np.ones(img.shape[:2], dtype=np.float32)
                    else:
                        logger.warning(
                            f"[Cinema Read] Cannot load EXR '{os.path.basename(fpath)}'. "
                            f"Install OpenEXR (pip install OpenEXR) or imageio for EXR support."
                        )
                        continue
                except Exception as e:
                    logger.warning(f"[Cinema Read] EXR/HDR load failed for {fpath}: {e}")
                    continue

            if img is None:
                raw = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    logger.warning(f"[Cinema Read] Skipping unreadable file: {fpath}")
                    continue
                if raw.ndim == 2:
                    img     = np.stack([raw, raw, raw], axis=-1);  del raw
                    mask_np = np.ones(img.shape[:2], dtype=np.float32)
                elif raw.shape[-1] == 4:
                    mask_np = raw[..., 3].astype(np.float32) / (255.0 if raw.dtype == np.uint8 else 65535.0)
                    img     = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB);  del raw
                else:
                    img     = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB);  del raw
                    mask_np = np.ones(img.shape[:2], dtype=np.float32)

            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            else:
                img = img.astype(np.float32)

            frame_t = torch.from_numpy(img[:, :, :c_out].copy()).unsqueeze(0)
            del img
            frame_t = apply_input_transform(frame_t, input_colorspace)

            batch_out[loaded]  = frame_t.squeeze(0)
            batch_mask[loaded] = torch.from_numpy(mask_np) if mask_np is not None \
                                  else torch.ones(h0, w0, dtype=torch.float32)
            del frame_t, mask_np
            loaded += 1

        if loaded == 0:
            raise ValueError(f"[Cinema Read] No valid images loaded from: {source_path}")

        if loaded < n_frames:
            batch_out  = batch_out[:loaded]
            batch_mask = batch_mask[:loaded]

        seq_fps  = 24.0
        fps_file = os.path.join(folder, "fps.txt")
        if os.path.exists(fps_file):
            try:
                with open(fps_file) as f:
                    seq_fps = float(f.read().strip().split()[0])
            except (ValueError, OSError) as e:
                logger.debug(f"[Cinema Read] fps.txt parse failed: {e}")
        fps = fps_override if fps_override > 0 else seq_fps

        gc.collect()
        logger.info(f"[Cinema Read] Sequence: {loaded} frames "
                    f"'{folder}' ({w0}×{h0} @ {fps}fps)")
        return (batch_out, batch_mask, loaded, w0, h0, float(fps), None, source_path)

# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE: DIGITAL CINEMA WRITE (UNIVERSAL)
# ═══════════════════════════════════════════════════════════════════════════════

WRITE_FORMATS = [
    "Image Sequence — EXR (32-bit)",
    "Image Sequence — Radiance HDR (.hdr)",
    "Image Sequence — PNG (16-bit)",
    "Image Sequence — PNG (8-bit)",
    "Image Sequence — TIFF (32-bit Float)",
    "Image Sequence — TIFF (16-bit)",
    "Image Sequence — JPEG",
    # GIF/WEBP animated belong in Sequence mode; static variants in Single Image.
    # They have no audio track and are not cinema formats.
    "GIF — Animated", "WEBP — Animated",
    "Video — MP4 (H.264)",
    "Video — MP4 (H.265 10-bit)",
    "Video — MOV (ProRes 422 HQ)",
    "Video — MOV (ProRes 4444)",
    "Video — MOV (ProRes 4444 XQ)",
    "Video — MOV (ProRes 4444 HDR Log)",
    "Video — MOV (DNxHR HQ)",
    "Video — MOV (DNxHR 444)",
    # Single Image mode uses shorter labels (no "Image Sequence —" prefix).
    # The JS FORMAT_GROUPS["Single Image"] references these values.
    "EXR (32-bit)",
    "Radiance HDR (.hdr)",
    "PNG (16-bit)",
    "PNG (8-bit)",
    "TIFF (32-bit Float)",
    "TIFF (16-bit)",
    "JPEG",
    "GIF",
    "WEBP",
]

# BUG 5 FIX: "None" renamed to "Uncompressed" — OpenEXR expects "NO_COMPRESSION"
# (uppercase). _norm_compression() normalises both "Uncompressed" and legacy "None".
COMPRESSIONS = ["ZIP", "ZIPS", "PIZ", "RLE", "Uncompressed", "PXR24", "B44", "B44A", "DWAA", "DWAB"]
BIT_DEPTHS = ["16-bit Half Float", "32-bit Float"]
ALPHA_MODES = ["None", "From Image", "Solid White", "Solid Black"]
# Audio export format options for image sequence / single image write modes.
# Lossless-only formats — no lossy codecs in a VFX pipeline.
AUDIO_EXPORT_FORMATS = [
    "None",
    "WAV — PCM 32-bit Float",
    "WAV — PCM 24-bit",
    "WAV — PCM 16-bit",
    "AIFF — PCM 24-bit",
    "FLAC — Lossless",
]

# BUG 5 FIX helper — normalise the compression string before it reaches OpenEXR.
# UI sends "Uncompressed"; legacy workflows may have "None". Both map to "NO_COMPRESSION".
def _norm_compression(comp: str) -> str:
    if comp in ("Uncompressed", "None", "NONE", "none"):
        return "NO_COMPRESSION"
    return comp

# ALBABIT-FIX: Build a PIL PngInfo object from a metadata dict for tEXt chunk embedding.
def _build_pnginfo(meta: dict):
    if not _HAS_PIL:
        return None
    info = PNGInfo()
    for k, v in meta.items():
        info.add_text(str(k), str(v))
    return info

# ───────────────────────────────────────────────────────────────────────────────
# Network / Remote output helper
# ───────────────────────────────────────────────────────────────────────────────

def _copy_to_remote(local_path: str, remote_path: str) -> bool:
    """
    Copy a local file or directory to a remote path.
    Supports UNC paths (\\\\server\\share\\path), S3 URIs (s3://bucket/key),
    and local absolute paths (shutil fallback).
    """
    import shutil
    rp = remote_path.strip()
    if not rp:
        return False

    if rp.startswith("s3://"):
        try:
            import boto3
            s3 = boto3.client("s3")
            parts  = rp[5:].split("/", 1)
            bucket = parts[0]
            key    = (parts[1] if len(parts) > 1 else "").rstrip("/")
            import pathlib
            local_p = str(local_path)
            if pathlib.Path(local_p).is_dir():
                for fp in pathlib.Path(local_p).rglob("*"):
                    if fp.is_file():
                        rel    = fp.relative_to(local_p)
                        s3_key = f"{key}/{rel}".replace("\\", "/")
                        s3.upload_file(str(fp), bucket, s3_key)
                        logger.info(f"[Remote] S3 upload: {fp} → s3://{bucket}/{s3_key}")
            else:
                fname  = pathlib.Path(local_p).name
                s3_key = f"{key}/{fname}" if key else fname
                s3.upload_file(local_p, bucket, s3_key)
                logger.info(f"[Remote] S3 upload: {local_p} → s3://{bucket}/{s3_key}")
            return True
        except ImportError:
            logger.warning("[Remote] boto3 not installed. Install via: pip install boto3")
            return False
        except Exception as e:
            logger.error(f"[Remote] S3 upload failed: {e}")
            return False

    try:
        import pathlib, shutil as _sh
        src = pathlib.Path(str(local_path))
        dst = pathlib.Path(rp)
        dst.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            _sh.copytree(str(src), str(dst / src.name), dirs_exist_ok=True)
        else:
            _sh.copy2(str(src), str(dst / src.name))
        logger.info(f"[Remote] Copied {src} → {dst}")
        return True
    except Exception as e:
        logger.error(f"[Remote] Network copy failed: {e}")
        return False


class RadianceDigitalCinemaWrite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (image_video_type,),
                "filename_prefix": ("STRING", {"default": "◎ Radiance"}),
                "write_mode": (["Video", "Sequence", "Single Image"], {"default": "Video"}),
                "output_format": (WRITE_FORMATS, {"default": "Video — MP4 (H.265 10-bit)"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0}),
                "quality": ("INT", {
                    "default": 80, "min": 0, "max": 100,
                    "tooltip": (
                        "Output quality — behavior depends on format:\n"
                        "• H.264 / H.265: maps to CRF (0=worst/smallest, 100=lossless). Default 80 = CRF ~10.\n"
                        "• JPEG sequences: 0–100 direct quality scale (0=worst, 100=best).\n"
                        "• WEBP: 0–100 direct quality scale.\n"
                        "• ProRes / PNG / EXR / HDR: no effect (lossless or fixed-bitrate codec)."
                    ),
                }),
                "output_color_space": (INPUT_COLORSPACES, {
                    "default": "sRGB (Standard)",
                    "tooltip": (
                        "Color space transform applied to the image before writing to disk.\n"
                        "• sRGB (Standard): standard display delivery (web, SDR monitors).\n"
                        "• Linear (sRGB): raw linear light, no gamma curve — for compositing pipelines.\n"
                        "• ACEScg: wide-gamut linear VFX working space (DaVinci Resolve, Nuke).\n"
                        "• DaVinci Intermediate / ARRI LogC3-4 / Sony S-Log3 / V-Log: log encoding "
                        "for roundtrip with camera footage or grading in a log-native timeline.\n"
                        "• ACEScct: log-like tone curve for ACES grading workflows."
                    ),
                }),
                "broadcast_safe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Clamps output values to [0.0–1.0] and applies a filmic ACES tone curve. "
                        "Only active when output_color_space is set to 'sRGB (Standard)'.\n"
                        "• True: safe for SDR delivery — prevents clipping artifacts on display.\n"
                        "• False: values above 1.0 are preserved — required for HDR and VFX pipelines "
                        "(e.g. EXR export to DaVinci Resolve with inverse tonemapping)."
                    ),
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Output directory for saved files.\n"
                        "• Empty: uses ComfyUI's default output folder.\n"
                        "• Relative path (e.g. 'MyProject/Renders'): subfolder created inside the "
                        "ComfyUI output directory. Do NOT start with '/' or '\\'.\n"
                        "• Absolute path (e.g. 'D:/VFX/Shots/sh010'): writes directly to that location.\n"
                        "Forward slashes and backslashes are both accepted. "
                        "Paths wrapped in quotes (e.g. \"D:\\Renders\") are also accepted."
                    ),
                }),
                "remote_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional remote output path. Supports:\n"
                        "  UNC:  \\\\server\\share\\folder\n"
                        "  S3:   s3://bucket/prefix\n"
                        "Leave empty to write locally only. "
                        "Paths wrapped in quotes are accepted."
                    ),
                }),
                "start_frame": ("INT", {"default": 1, "min": 0}),
                "bit_depth": (BIT_DEPTHS, {"default": "32-bit Float"}),
                "compression": (COMPRESSIONS, {
                    "default": "ZIP",
                    "tooltip": (
                        "EXR compression method — applies only to EXR sequences and single images.\n"
                        "• ZIP: lossless, multi-scanline — recommended for general VFX use.\n"
                        "• ZIPS: lossless, single-scanline ZIP — faster but larger files.\n"
                        "• PIZ: lossless wavelet — best for grainy or noisy images.\n"
                        "• RLE: lossless run-length — fast for flat/CG renders.\n"
                        "• Uncompressed: no compression — fastest I/O, largest files.\n"
                        "• PXR24: lossy 24-bit float (Pixar legacy format).\n"
                        "• B44 / B44A: lossy fixed-ratio — fast decode, good for playback.\n"
                        "• DWAA / DWAB: lossy DCT-based — smallest files, some quality loss.\n"
                        "Has no effect on PNG, JPEG, HDR, or video formats."
                    ),
                }),
                "alpha_mode": (ALPHA_MODES, {
                    "default": "From Image",
                    "tooltip": (
                        "Alpha channel handling for formats that support transparency.\n"
                        "• None: discard alpha — export as RGB / RGB-only.\n"
                        "• From Image: use alpha channel from the source image if present.\n"
                        "• Solid White: force alpha = 1.0 (fully opaque).\n"
                        "• Solid Black: force alpha = 0.0 (fully transparent).\n"
                        "Supported formats: EXR (RGBA float), PNG 8-bit (RGBA via PIL), PNG 16-bit (RGBA via cv2).\n"
                        "Not applicable to TIFF, JPEG, Radiance HDR, or video formats."
                    ),
                }),
                "frame_padding": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Zero-padding width for frame numbers (e.g. 4 → 0001, 6 → 000001)."}),
                # Audio export widgets placed before custom_metadata to avoid overlap
                # with the multiline text area when shown/hidden by the JS.
                "write_external_audio_file": (AUDIO_EXPORT_FORMATS, {
                    "default": "None",
                    "tooltip": (
                        "Export a separate audio file alongside the output. "
                        "In Video mode, audio is always embedded in the container — "
                        "this additionally exports a lossless copy. "
                        "In Sequence or Single Image mode, this is the only way to preserve the audio track."
                    ),
                }),
                "audio_filename_suffix": ("STRING", {
                    "default": "_audio",
                    "tooltip": (
                        "Suffix added to filename_prefix for the exported audio file. "
                        "The file is saved in the same output folder as the image/video "
                        "(e.g. prefix + '_audio' → 'MyShot_audio.wav')."
                    ),
                }),
                "custom_metadata": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "Custom key-value metadata embedded in the output file.\n"
                        "Enter one entry per line in 'key=value' format "
                        "(e.g. 'shot=sh010' or 'colorspace=ACEScg').\n"
                        "• EXR: written as named EXR header attributes.\n"
                        "• PNG 8-bit: written as tEXt chunks via PIL.\n"
                        "• PNG 16-bit: metadata not written (cv2 limitation — use EXR for full metadata support).\n"
                        "• Other formats: metadata may be silently ignored depending on container support."
                    ),
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }


    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "write"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/IO"

    def write(self, **kwargs):
        RadianceWrite().write(**kwargs)
        return {}

class RadianceWrite:
    def write(self, image, filename_prefix, write_mode="Video", output_format="", fps=24.0, quality=80,
              output_color_space="sRGB (Standard)", broadcast_safe=False, audio=None,
              output_path="", remote_path="", start_frame=1, bit_depth="32-bit Float", compression="ZIP",
              alpha_mode="From Image", frame_padding=4, custom_metadata="",
              write_external_audio_file="None", audio_filename_suffix="_audio",
              prompt=None, extra_pnginfo=None):

        output_path  = _strip_path_quotes(output_path)
        remote_path  = _strip_path_quotes(remote_path)

        if hasattr(image, "get_components"): image = image.get_components().images
        elif isinstance(image, dict) and "samples" in image: image = image["samples"]

        if not isinstance(image, torch.Tensor):
            vpath = find_video_path(image)
            image = _load_video_frames(vpath) if vpath else None

        if image is None: raise ValueError("No image/video data")

        full_out   = get_safe_output_dir(folder_paths.get_output_directory(), output_path, allow_absolute=True)
        images_out = apply_output_transform(image, output_color_space, broadcast_safe)
        images_np  = images_out.cpu().numpy()
        ts         = int(time.time())

        # Build Metadata
        meta = {
            "software": "Radiance v2.3",
            "created": datetime.datetime.now().isoformat(),
            "colorspace": output_color_space,
            "hdr_integrity": "Unclamped Linear" if "Linear" in output_color_space else "Log-Encoded Carrier",
            "broadcast_safe": str(broadcast_safe),
        }
        if custom_metadata:
            for line in custom_metadata.strip().split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    meta[k.strip()] = v.strip()

        # GIF/WEBP animated → _write_animated (Sequence mode)
        # GIF/WEBP static   → _write_animated with is_single_image=True (Single Image mode)
        # GIF/WEBP are not cinema formats; they are NOT routed through _write_video.
        is_animated_fmt = "GIF" in output_format or "WEBP" in output_format
        if write_mode == "Video" and "Video" in output_format:
            res = self._write_video(images_np, filename_prefix, output_format, fps, quality, output_color_space, full_out, ts, audio, broadcast_safe)
        elif is_animated_fmt and write_mode == "Sequence":
            res = self._write_animated(images_np, filename_prefix, output_format, output_dir=full_out, ts=ts, fps=fps, quality=quality, is_single_image=False)
        elif is_animated_fmt and write_mode == "Single Image":
            res = self._write_animated(images_np[:1], filename_prefix, output_format, output_dir=full_out, ts=ts, fps=fps, quality=quality, is_single_image=True)
        elif write_mode == "Single Image":
            res = self._write_sequence(images_np[:1], filename_prefix, output_format, quality, full_out, ts, start_frame, frame_padding, False, bit_depth, compression, meta, alpha_mode, image, is_single_image=True)
        else:
            res = self._write_sequence(images_np, filename_prefix, output_format, quality, full_out, ts, start_frame, frame_padding, True, bit_depth, compression, meta, alpha_mode, image)

        if remote_path:
            if res and os.path.exists(res):
                _copy_to_remote(res, remote_path)
            else:
                logger.warning(
                    f"[Cinema Write] Remote copy skipped — output path does not exist: '{res}'. "
                    f"Check that the write completed successfully before enabling remote_path."
                )

        # Export audio after images so we can resolve the correct target directory.
        # In Sequence mode, _write_sequence returns the sequence subfolder (prefix_timestamp/) —
        # the audio file should land there alongside the frames, not in the parent output_path.
        if write_external_audio_file != "None" and audio is not None:
            audio_dir = res if write_mode == "Sequence" and os.path.isdir(res) else full_out
            self._write_audio_file(audio, write_external_audio_file, filename_prefix, audio_filename_suffix, audio_dir, ts)

        return (image, res, res)

    def _write_video(self, images_np, prefix, fmt, fps, quality, color_space, output_dir, ts, audio, broadcast_safe):
        import imageio.v3 as iio

        fpath_mp4  = os.path.join(output_dir, f"{prefix}_{ts}.mp4")
        fpath_mov  = os.path.join(output_dir, f"{prefix}_{ts}.mov")
        fpath_gif  = os.path.join(output_dir, f"{prefix}_{ts}.gif")
        fpath_webp = os.path.join(output_dir, f"{prefix}_{ts}.webp")

        if "H.264" in fmt:
            # ── H.264 / AVC — 8-bit, ffmpeg subprocess with CRF quality control ──
            # CRF range for H.264: 0 (lossless) – 51 (worst). quality 0-100 → CRF.
            crf_h264  = max(0, min(51, int((1.0 - quality / 100.0) * 51)))
            frames_u8 = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
            h, w      = frames_u8.shape[1], frames_u8.shape[2]
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", str(crf_h264),
                "-preset", "slow",
                "-movflags", "+faststart",
                fpath_mp4,
            ]
            raw = frames_u8.tobytes()
            result = subprocess.run(cmd, input=raw, capture_output=True, timeout=600)  # nosec B603
            if result.returncode != 0:
                logger.error(
                    f"[RadianceWrite] H.264 encode failed:\n"
                    f"{result.stderr.decode(errors='replace')}"
                )
                raise RuntimeError("ffmpeg H.264 encode failed — see log for details")
            fpath = fpath_mp4

        elif "H.265" in fmt:
            # ── H.265 / HEVC — true 10-bit, ffmpeg subprocess ─────────────────
            # CRF range for H.265: 0 (lossless) – 51 (worst). quality 0-100 → CRF.
            crf_h265  = max(0, min(51, int((1.0 - quality / 100.0) * 51)))
            frames_u8 = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
            h, w      = frames_u8.shape[1], frames_u8.shape[2]
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vcodec", "libx265",
                "-pix_fmt", "yuv420p10le",
                "-crf", str(crf_h265),
                "-preset", "slow",
                "-tag:v", "hvc1",
                "-movflags", "+faststart",
                fpath_mp4,
            ]
            raw = frames_u8.tobytes()
            result = subprocess.run(cmd, input=raw, capture_output=True, timeout=600)  # nosec B603
            if result.returncode != 0:
                logger.error(
                    f"[RadianceWrite] H.265 encode failed:\n"
                    f"{result.stderr.decode(errors='replace')}"
                )
                raise RuntimeError("ffmpeg H.265 encode failed — see log for details")
            fpath = fpath_mp4

        elif "ProRes 422" in fmt:
            # ── ProRes 422 HQ — 10-bit 4:2:2, prores_ks profile 3 ─────────────
            # BUG 4 FIX: was silently falling through to H.264 else-branch because
            # "ProRes 422 HQ" does not match "ProRes 4444". Now has an explicit branch.
            frames_u16 = (np.clip(images_np, 0, 1) * 65535).astype(np.uint16)
            h, w       = frames_u16.shape[1], frames_u16.shape[2]
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "rgb48le",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vcodec", "prores_ks",
                "-profile:v", "3",
                "-pix_fmt", "yuv422p10le",
                "-vendor", "apl0",
                fpath_mov,
            ]
            raw = frames_u16.tobytes()
            result = subprocess.run(cmd, input=raw, capture_output=True, timeout=600)  # nosec B603
            if result.returncode != 0:
                logger.error(
                    f"[RadianceWrite] ProRes 422 HQ encode failed:\n"
                    f"{result.stderr.decode(errors='replace')}"
                )
                raise RuntimeError("ffmpeg ProRes 422 HQ encode failed — see log for details")
            fpath = fpath_mov

        elif "ProRes 4444" in fmt:
            # ── ProRes 4444 / 4444 XQ / 4444 HDR Log — 12-bit 4:4:4 ──────────
            # profile 4 = 4444, profile 5 = 4444 XQ (higher data rate).
            # "HDR Log" is a workflow label; log encoding is applied upstream by
            # output_color_space. No additional processing at codec level.
            is_xq      = "XQ" in fmt
            frames_u16 = (np.clip(images_np, 0, 1) * 65535).astype(np.uint16)
            h, w       = frames_u16.shape[1], frames_u16.shape[2]
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "rgb48le",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vcodec", "prores_ks",
                "-profile:v", "5" if is_xq else "4",
                "-pix_fmt", "yuv444p12le",
                "-vendor", "apl0",
                fpath_mov,
            ]
            raw = frames_u16.tobytes()
            result = subprocess.run(cmd, input=raw, capture_output=True, timeout=600)  # nosec B603
            if result.returncode != 0:
                logger.error(
                    f"[RadianceWrite] ProRes 4444 encode failed:\n"
                    f"{result.stderr.decode(errors='replace')}"
                )
                raise RuntimeError("ffmpeg ProRes 4444 encode failed — see log for details")
            fpath = fpath_mov

        elif "DNxHR" in fmt:
            # ── DNxHR — Avid / Resolve editor codec, .mov container ─────────────
            # DNxHR HQ  = 4:2:2 10-bit — editorial, broadcast
            # DNxHR 444 = 4:4:4 12-bit — VFX, finishing
            is_444  = "444" in fmt
            profile = "dnxhr_444" if is_444 else "dnxhr_hq"
            pix_fmt = "yuv444p12le" if is_444 else "yuv422p10le"
            frames_u16 = (np.clip(images_np, 0, 1) * 65535).astype(np.uint16)
            h, w       = frames_u16.shape[1], frames_u16.shape[2]
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "rgb48le",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vcodec", "dnxhd",
                "-profile:v", profile,
                "-pix_fmt", pix_fmt,
                "-vendor", "appl",
                fpath_mov,
            ]
            raw = frames_u16.tobytes()
            result = subprocess.run(cmd, input=raw, capture_output=True, timeout=600)  # nosec B603
            if result.returncode != 0:
                logger.error(
                    f"[RadianceWrite] DNxHR encode failed:\n"
                    f"{result.stderr.decode(errors='replace')}"
                )
                raise RuntimeError("ffmpeg DNxHR encode failed — see log for details")
            fpath = fpath_mov

        elif "GIF" in fmt:
            # Fallback for GIF in Video mode (normally routed via _write_animated)
            data = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
            iio.imwrite(fpath_gif, data, duration=1.0 / max(fps, 1e-3), loop=0)
            fpath = fpath_gif

        elif "WEBP" in fmt:
            # Fallback for WEBP in Video mode (normally routed via _write_animated)
            webp_q    = max(0, min(100, int(quality)))
            frames_u8 = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
            h, w      = frames_u8.shape[1], frames_u8.shape[2]
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vcodec", "libwebp_anim",
                "-loop", "0",
                "-lossless", "0",
                "-compression_level", "6",
                "-q:v", str(webp_q),
                fpath_webp,
            ]
            raw = frames_u8.tobytes()
            result = subprocess.run(cmd, input=raw, capture_output=True, timeout=600)  # nosec B603
            if result.returncode != 0:
                logger.error(
                    f"[RadianceWrite] WEBP encode failed:\n"
                    f"{result.stderr.decode(errors='replace')}"
                )
                raise RuntimeError("ffmpeg WEBP encode failed — see log for details")
            fpath = fpath_webp

        else:
            raise ValueError(
                f"[RadianceWrite] Unrecognised video output_format: {fmt!r}. "
                f"Expected one of the entries in WRITE_FORMATS."
            )

        if audio:
            if fpath.lower().endswith((".gif", ".webp")):
                logger.info("[Cinema Write] Skipping audio mux for GIF/WEBP (container does not support audio).")
            else:
                self._mux_audio(fpath, audio)
        return fpath

    def _write_animated(self, images_np, prefix, fmt, output_dir, ts, fps=24.0, quality=80, is_single_image=False):
        import imageio.v3 as iio
        try:
            is_gif      = "GIF" in fmt
            ext         = ".gif" if is_gif else ".webp"
            fname       = f"{prefix}{ext}" if is_single_image else f"{prefix}_{ts}{ext}"
            fpath       = os.path.join(output_dir, fname)
            os.makedirs(output_dir, exist_ok=True)
            frames_u8   = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
            duration_ms = int(1000.0 / max(fps, 1.0))
            n_frames    = len(frames_u8)
            if is_gif:
                iio.imwrite(fpath, frames_u8, duration=duration_ms, loop=0)
            else:
                iio.imwrite(fpath, frames_u8, duration=duration_ms, loop=0, quality=int(quality))
            label = "GIF" if is_gif else "WEBP"
            logger.info(f"◎ {label} written: {fpath} ({n_frames} frame{'s' if n_frames != 1 else ''})")
            return fpath
        except Exception as e:
            logger.error(f"[RadianceWrite] Animated write failed ({fmt}): {e}")
            return ""

    def _write_audio_file(self, audio, fmt, prefix, suffix, output_dir, ts):
        try:
            import struct
            waveform = audio.get("waveform")
            sr = audio.get("sample_rate", 44100)
            if waveform is None: return

            wav_np     = waveform.squeeze(0).cpu().numpy()
            n_channels = wav_np.shape[0]
            os.makedirs(output_dir, exist_ok=True)
            raw_f32 = wav_np.T.flatten().astype(np.float32).tobytes()

            if "32-bit Float" in fmt:
                data_size   = len(raw_f32)
                block_align = n_channels * 4
                wav_path    = os.path.join(output_dir, f"{prefix}_{ts}{suffix}.wav")
                with open(wav_path, "wb") as f:
                    f.write(b"RIFF" + struct.pack("<I", 36 + data_size) +
                            b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, n_channels, sr,
                            sr * block_align, block_align, 32) +
                            b"data" + struct.pack("<I", data_size) + raw_f32)
                logger.info(f"◎ Audio exported: {wav_path}")
                return

            if "WAV" in fmt and "24-bit" in fmt:
                out_path = os.path.join(output_dir, f"{prefix}_{ts}{suffix}.wav")
                codec    = "pcm_s24le"
            elif "WAV" in fmt and "16-bit" in fmt:
                out_path = os.path.join(output_dir, f"{prefix}_{ts}{suffix}.wav")
                codec    = "pcm_s16le"
            elif "AIFF" in fmt:
                out_path = os.path.join(output_dir, f"{prefix}_{ts}{suffix}.aiff")
                codec    = "pcm_s24be"
            elif "FLAC" in fmt:
                out_path = os.path.join(output_dir, f"{prefix}_{ts}{suffix}.flac")
                codec    = "flac"
            else:
                logger.warning(f"[RadianceWrite] Unknown audio_export format: {fmt}")
                return

            cmd = [
                "ffmpeg", "-y",
                "-f", "f32le", "-ar", str(sr), "-ac", str(n_channels),
                "-i", "pipe:0",
                "-c:a", codec,
                out_path,
            ]
            result = subprocess.run(cmd, input=raw_f32, capture_output=True, timeout=120)  # nosec B603
            if result.returncode != 0:
                logger.error(f"[RadianceWrite] Audio export failed ({fmt}):\n{result.stderr.decode(errors='replace')}")
            else:
                logger.info(f"◎ Audio exported: {out_path}")
        except Exception as e:
            logger.error(f"[RadianceWrite] Failed to export audio ({fmt}): {e}")

    def _write_sequence(self, images_np, prefix, fmt, quality, output_dir, ts, start, padding, use_ts, bdepth, comp, meta, alpha_mode, images_tensor=None, is_single_image=False):
        # BUG 5 FIX: normalise compression string before it reaches write_exr_robust.
        comp = _norm_compression(comp)

        if is_single_image:
            target = output_dir
            if "EXR" in fmt: ext = ".exr"
            elif "HDR" in fmt: ext = ".hdr"
            elif "JPEG" in fmt: ext = ".jpg"
            elif "TIFF" in fmt: ext = ".tif"
            else: ext = ".png"

            if os.path.exists(os.path.join(target, f"{prefix}{ext}")):
                v_prefix = f"{prefix}_v"
                idx = get_next_index(target, v_prefix, ext, 1)
                if idx == 0: idx = 2
                prefix = f"{v_prefix}{idx}"
        else:
            target = os.path.join(output_dir, f"{prefix}_{ts}") if (use_ts and len(images_np) > 1) else output_dir

        os.makedirs(target, exist_ok=True)

        paths = []
        for i, frame in enumerate(images_np):
            num        = str(start + i).zfill(padding)
            frame_meta = {**meta, "frame": start + i}

            # ALBABIT-FIX: Extract alpha once per frame — shared by EXR and PNG paths.
            alpha_np = None
            if alpha_mode != "None" and images_tensor is not None:
                t = images_tensor[i] if images_tensor.dim() == 4 else images_tensor
                if t.shape[-1] == 4:
                    if alpha_mode == "From Image":
                        alpha_np = t[..., 3].cpu().numpy().astype(np.float32)
                    elif alpha_mode == "Solid White":
                        alpha_np = np.ones(frame.shape[:2], dtype=np.float32)
                    elif alpha_mode == "Solid Black":
                        alpha_np = np.zeros(frame.shape[:2], dtype=np.float32)
                elif alpha_mode in ("Solid White", "Solid Black"):
                    alpha_np = np.ones(frame.shape[:2], dtype=np.float32) if alpha_mode == "Solid White" else np.zeros(frame.shape[:2], dtype=np.float32)

            if "EXR" in fmt:
                ext   = ".exr"
                fname = f"{prefix}{ext}" if is_single_image else f"{prefix}.{num}{ext}"
                fpath = os.path.join(target, fname)

                if write_exr_robust:
                    success = False
                    if alpha_np is not None:
                        try:
                            success = write_exr_robust(fpath, frame, bdepth, comp, frame_meta, alpha=alpha_np)
                        except TypeError:
                            logger.debug("[Cinema Write] write_exr_robust does not accept alpha kwarg — writing RGB only")
                            success = write_exr_robust(fpath, frame, bdepth, comp, frame_meta)
                    else:
                        success = write_exr_robust(fpath, frame, bdepth, comp, frame_meta)
                    if not success:
                        raise RuntimeError(
                            f"[Cinema Write] write_exr_robust returned False for '{fpath}'. "
                            f"Install OpenEXR (pip install OpenEXR) for reliable EXR output."
                        )
                else:
                    raise RuntimeError(
                        f"[Cinema Write] EXR output requires OpenEXR. "
                        f"Install with: pip install OpenEXR\nAttempted path: '{fpath}'"
                    )

            elif "HDR" in fmt:
                ext   = ".hdr"
                fname = f"{prefix}{ext}" if is_single_image else f"{prefix}.{num}{ext}"
                fpath = os.path.join(target, fname)
                if write_hdr_rgbe: write_hdr_rgbe(fpath, frame)
                else: cv2.imwrite(fpath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32))

            elif "JPEG" in fmt:
                ext   = ".jpg"
                fname = f"{prefix}{ext}" if is_single_image else f"{prefix}.{num}{ext}"
                fpath = os.path.join(target, fname)
                jpeg_q = max(0, min(100, int(quality)))
                cv2.imwrite(fpath, cv2.cvtColor((np.clip(frame, 0, 1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])

            elif "TIFF (32-bit Float)" in fmt:
                ext   = ".tif"
                fname = f"{prefix}{ext}" if is_single_image else f"{prefix}.{num}{ext}"
                fpath = os.path.join(target, fname)
                ok = cv2.imwrite(fpath, cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2BGR))
                if not ok:
                    logger.error(f"[Cinema Write] TIFF 32-bit write failed for: {fpath}")

            elif "TIFF (16-bit)" in fmt:
                ext   = ".tif"
                fname = f"{prefix}{ext}" if is_single_image else f"{prefix}.{num}{ext}"
                fpath = os.path.join(target, fname)
                ok = cv2.imwrite(fpath, cv2.cvtColor((np.clip(frame, 0, 1) * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR))
                if not ok:
                    logger.error(f"[Cinema Write] TIFF 16-bit write failed for: {fpath}")

            else:  # PNG
                ext    = ".png"
                fname  = f"{prefix}{ext}" if is_single_image else f"{prefix}.{num}{ext}"
                fpath  = os.path.join(target, fname)
                is_8bit = "8-bit" in fmt
                data_f  = np.clip(frame, 0, 1)
                if is_8bit and _HAS_PIL:
                    # ALBABIT-FIX: 8-bit PNG — PIL writes tEXt metadata chunks and supports RGBA.
                    data_u8 = (data_f * 255).astype(np.uint8)
                    pnginfo = _build_pnginfo(frame_meta)
                    if alpha_np is not None:
                        alpha_u8 = (np.clip(alpha_np, 0, 1) * 255).astype(np.uint8)
                        pil_img  = PILImage.fromarray(np.dstack([data_u8, alpha_u8]), "RGBA")
                    else:
                        pil_img  = PILImage.fromarray(data_u8, "RGB")
                    pil_img.save(fpath, pnginfo=pnginfo)
                else:
                    # ALBABIT-FIX: 16-bit PNG — cv2 handles uint16 BGRA natively.
                    # tEXt metadata is not written (cv2 limitation); use 8-bit or EXR for embedded metadata.
                    scale    = 255 if is_8bit else 65535
                    dtype    = np.uint8 if is_8bit else np.uint16
                    data_px  = (data_f * scale).astype(dtype)
                    if alpha_np is not None:
                        alpha_px = (np.clip(alpha_np, 0, 1) * scale).astype(dtype)
                        data_out = cv2.cvtColor(np.dstack([data_px, alpha_px]).astype(dtype), cv2.COLOR_RGBA2BGRA)
                    else:
                        data_out = cv2.cvtColor(data_px, cv2.COLOR_RGB2BGR)
                    ok = cv2.imwrite(fpath, data_out)
                    if not ok:
                        logger.error(f"[Cinema Write] cv2.imwrite failed for: {fpath}")
            paths.append(fpath)

        if not paths:
            raise RuntimeError(
                f"[Cinema Write] No frames written to {target!r}. "
                f"Check that images_np is not empty and the output format is valid."
            )
        return paths[0] if len(paths) == 1 else target

    def _mux_audio(self, video_path, audio):
        # BUG 3 FIX: use try/finally so tmp_wav is always cleaned up even on exception.
        # BUG 3 FIX: use pcm_s24le for .mov (ProRes) — AAC in .mov breaks Avid/broadcast QC.
        tmp_wav = None
        try:
            waveform, sr = audio.get("waveform"), audio.get("sample_rate", 44100)
            if waveform is None: return
            import struct
            wav_np = waveform.squeeze(0).cpu().numpy()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = tmp.name
            with open(tmp_wav, "wb") as f:
                f.write(
                    b"RIFF" + struct.pack("<I", 36 + wav_np.size * 4) +
                    b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, wav_np.shape[0], sr,
                        sr * wav_np.shape[0] * 4, wav_np.shape[0] * 4, 32) +
                    b"data" + struct.pack("<I", wav_np.size * 4) +
                    wav_np.T.flatten().astype(np.float32).tobytes()
                )
            tmp_out     = video_path + ".tmp" + os.path.splitext(video_path)[1]
            is_mov      = video_path.lower().endswith(".mov")
            audio_codec = "pcm_s24le" if is_mov else "aac"
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-i", tmp_wav,
                 "-c:v", "copy", "-c:a", audio_codec, "-shortest", tmp_out],
                capture_output=True
            )
            if os.path.exists(tmp_out):
                os.replace(tmp_out, video_path)
        except Exception as e:
            logger.warning(f"[Cinema Write] Audio mux failed (video saved without audio): {e}")
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.unlink(tmp_wav)
                except OSError:
                    pass

# ───────────────────────────────────────────────────────────────────────────────
#                NODE: EXR MULTI-PART WRITER
# ───────────────────────────────────────────────────────────────────────────────

class RadianceEXRMultiPart:
    """
    ◎ Radiance EXR Multi-Part

    Write a named multi-part EXR v2 file combining up to 6 AOV layers
    (Arbitrary Output Variables) into a single file readable by Nuke,
    DaVinci Resolve (Flatten Layers), and Fusion.

    Standard part names and their channel assignments:
      • beauty      → R, G, B  (or R, G, B, A if alpha connected)
      • depth       → Z  (single channel)
      • normal      → NX, NY, NZ
      • albedo      → albedo.R, albedo.G, albedo.B
      • custom_1/2  → custom_1.R, custom_1.G, custom_1.B

    Fallback: If OpenEXR v2 multi-part is unavailable, the node
    writes separate .beauty.exr / .depth.exr / etc. files instead.
    """

    CATEGORY = "FXTD Studios/Radiance/IO"
    FUNCTION = "write_multipart"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "radiance_multipart"}),
                "beauty": ("IMAGE", {"tooltip": "Beauty pass — R, G, B (or RGBA)."}),
                "bit_depth": (BIT_DEPTHS, {"default": "16-bit Half Float"}),
                "compression": (COMPRESSIONS, {"default": "ZIP"}),
            },
            "optional": {
                "depth": ("IMAGE", {"tooltip": "Depth / Z pass — single channel or first channel used."}),
                "normal": ("IMAGE", {"tooltip": "World-space normals — NX, NY, NZ."}),
                "albedo": ("IMAGE", {"tooltip": "Albedo / diffuse colour pass."}),
                "custom_1": ("IMAGE", {"tooltip": "Custom AOV 1."}),
                "custom_1_name": ("STRING", {"default": "emission"}),
                "custom_2": ("IMAGE", {"tooltip": "Custom AOV 2."}),
                "custom_2_name": ("STRING", {"default": "specular"}),
                "output_path": ("STRING", {"default": "",
                    "tooltip": "Output directory for saved EXR files. Absolute or relative to ComfyUI output. Paths wrapped in quotes are accepted."}),
                "remote_path": ("STRING", {"default": "",
                    "tooltip": "Optional S3 or UNC remote path. Paths wrapped in quotes are accepted."}),
                "frame_index": ("INT", {"default": 1, "min": 1}),
                "custom_metadata": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Optional key=value metadata lines embedded in EXR header."}),
            },
        }

    def write_multipart(
        self,
        filename_prefix: str,
        beauty: torch.Tensor,
        bit_depth: str = "16-bit Half Float",
        compression: str = "ZIP",
        depth: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None,
        albedo: Optional[torch.Tensor] = None,
        custom_1: Optional[torch.Tensor] = None,
        custom_1_name: str = "emission",
        custom_2: Optional[torch.Tensor] = None,
        custom_2_name: str = "specular",
        output_path: str = "",
        remote_path: str = "",
        frame_index: int = 1,
        custom_metadata: str = "",
    ) -> Tuple[str]:

        output_path = _strip_path_quotes(output_path)
        remote_path = _strip_path_quotes(remote_path)

        out_dir = get_safe_output_dir(
            folder_paths.get_output_directory(), output_path, allow_absolute=True
        )
        os.makedirs(out_dir, exist_ok=True)

        frame_num = str(frame_index).zfill(4)
        filepath  = os.path.join(out_dir, f"{filename_prefix}.{frame_num}.exr")

        compression = _norm_compression(compression)

        import re as _re
        _SAFE_PART_RE = _re.compile(r'^[A-Za-z0-9_\-\.]+$')
        for part_name in [custom_1_name, custom_2_name]:
            if part_name and not _SAFE_PART_RE.match(part_name):
                raise ValueError(
                    f"[EXRMultiPart] Invalid part name: '{part_name}'. "
                    f"EXR part names must contain only letters, digits, underscore, hyphen, or dot."
                )

        meta: Dict[str, Any] = {
            "software": "Radiance v2.3",
            "created": datetime.datetime.now().isoformat(),
        }
        for line in custom_metadata.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()

        def _to_np(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
            if t is None:
                return None
            arr = t.squeeze(0) if t.dim() == 4 and t.shape[0] == 1 else t
            if arr.dim() == 4:
                arr = arr[0]
            return arr.float().cpu().numpy()

        parts: Dict[str, Optional[np.ndarray]] = {}
        parts["beauty"]           = _to_np(beauty)
        if depth is not None:     parts["depth"]  = _to_np(depth)
        if normal is not None:    parts["normal"] = _to_np(normal)
        if albedo is not None:    parts["albedo"] = _to_np(albedo)
        if custom_1 is not None:  parts[custom_1_name or "custom_1"] = _to_np(custom_1)
        if custom_2 is not None:  parts[custom_2_name or "custom_2"] = _to_np(custom_2)

        parts = {k: v for k, v in parts.items() if v is not None}

        if write_exr_multipart is not None:
            success = write_exr_multipart(filepath, parts, bit_depth, compression, meta)
        else:
            logger.warning("[EXRMultiPart] write_exr_multipart not available — writing beauty-only EXR.")
            b_np    = parts.get("beauty")
            success = write_exr_robust(filepath, b_np, bit_depth, compression, meta) if (b_np is not None and write_exr_robust) else False

        if success:
            logger.info(f"[EXRMultiPart] Wrote {len(parts)} parts to: {filepath}")
        else:
            logger.error(f"[EXRMultiPart] Failed to write: {filepath}")
            raise RuntimeError(f"EXR multi-part write failed: {filepath}")

        if remote_path:
            _copy_to_remote(filepath, remote_path)

        return (filepath,)


NODE_CLASS_MAPPINGS = {
    "RadianceDigitalCinemaRead":  RadianceDigitalCinemaRead,
    "RadianceDigitalCinemaWrite": RadianceDigitalCinemaWrite,
    "RadianceEXRMultiPart":       RadianceEXRMultiPart,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceDigitalCinemaRead":  "◎ Radiance Read",
    "RadianceDigitalCinemaWrite": "◎ Radiance Write",
    "RadianceEXRMultiPart":       "◎ Radiance EXR Multi-Part",
}
