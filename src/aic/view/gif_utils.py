#!/usr/bin/env python
"""Shared GIF helpers for the view animators, so every generated GIF keeps a
CONSISTENT gradient/colouring: all frames are quantized against ONE 256-colour
palette (built from a montage of the frames), which removes the per-frame
GIF-palette flicker you get from Pillow's default per-frame quantization.

Usage:
    pal = shared_palette(rgb_frames)          # one palette for the whole set
    frames_p = [quantize(f, pal) for f in rgb_frames]
    save_gif(frames_p, path, duration=1000)
"""
from __future__ import annotations

from PIL import Image


def shared_palette(rgb_frames, colors: int = 256):
    """One adaptive palette covering ALL frames (montage + median-cut). Pass the
    complete set of frames that must share colours (e.g. every frame of every GIF
    in a batch), so the palette is identical everywhere."""
    if not rgb_frames:
        raise ValueError("no frames")
    fw, fh = rgb_frames[0].size
    montage = Image.new("RGB", (fw, fh * len(rgb_frames)))
    for i, f in enumerate(rgb_frames):
        montage.paste(f.convert("RGB"), (0, i * fh))
    return montage.quantize(colors=colors, method=Image.MEDIANCUT)


def quantize(frame, palette):
    """Map an RGB frame onto the shared palette (no dithering -> flat, stable colours)."""
    return frame.convert("RGB").quantize(palette=palette, dither=Image.NONE)


def save_gif(frames_p, path, duration: int = 1000, loop: int = 0):
    """Write palettized frames as a GIF (disposal=2 so each frame fully redraws;
    optimize=False so the shared palette is kept, not re-derived per frame)."""
    frames_p[0].save(path, save_all=True, append_images=frames_p[1:],
                     duration=duration, loop=loop, optimize=False, disposal=2)
