import os
from typing import TypeAlias, Mapping
from io import BytesIO

try:
    import pillow_jxl  # Ensure this is installed
except:
    from jxlpy import JXLImagePlugin
from PIL import ImageCms, Image, PngImagePlugin, ImageChops
from PIL.ImageCms import Intent
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

# Suppress the warning for large images
Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

# Color management profiles and intent flags
_SRGB = ImageCms.createProfile(colorSpace='sRGB')

IntentFlags: TypeAlias = Mapping[Intent, int]

_INTENT_FLAGS_INITIAL: IntentFlags = {
    Intent.PERCEPTUAL: ImageCms.FLAGS["HIGHRESPRECALC"],
    Intent.RELATIVE_COLORIMETRIC: ImageCms.FLAGS["HIGHRESPRECALC"] | ImageCms.FLAGS["BLACKPOINTCOMPENSATION"],
    Intent.SATURATION: ImageCms.FLAGS["HIGHRESPRECALC"],
    Intent.ABSOLUTE_COLORIMETRIC: ImageCms.FLAGS["HIGHRESPRECALC"]
}

_INTENT_FLAGS_FALLBACK: IntentFlags = {
    Intent.PERCEPTUAL: ImageCms.FLAGS["HIGHRESPRECALC"],
    Intent.RELATIVE_COLORIMETRIC: ImageCms.FLAGS["HIGHRESPRECALC"] | ImageCms.FLAGS["BLACKPOINTCOMPENSATION"],
    Intent.ABSOLUTE_COLORIMETRIC: ImageCms.FLAGS["HIGHRESPRECALC"]
}

def _coalesce_intent(intent: Intent | int) -> Intent:
    if isinstance(intent, Intent):
        return intent

    match intent:
        case 0:
            return Intent.PERCEPTUAL
        case 1:
            return Intent.RELATIVE_COLORIMETRIC
        case 2:
            return Intent.SATURATION
        case 3:
            return Intent.ABSOLUTE_COLORIMETRIC
        case _:
            raise ValueError("invalid ImageCms intent")

def open_srgb(
    fp,
    *,
    mode: str | None = "RGB",
    intent: Intent | int | None = Intent.RELATIVE_COLORIMETRIC,
    intent_flags: IntentFlags | None = None,
    intent_fallback: bool = True,
    formats: list[str] | tuple[str, ...] | None = None,
) -> Image.Image:
    img = Image.open(fp, formats=formats)

    if img.mode == 'P' and img.info.get('transparency'):
        img = img.convert('PA')

    if mode is None:
        match img.mode:
            case "RGBA" | "LA" | "PA":
                mode = "RGBA"
            case "RGBa" | "La":
                mode = "RGBa"
            case _:
                mode = "RGB"

    # ensure image is in sRGB color space
    if intent is not None:
        icc_raw = img.info.get("icc_profile")

        if icc_raw is not None:
            profile = ImageCms.ImageCmsProfile(BytesIO(icc_raw))
            intent = _coalesce_intent(intent)

            if img.mode == "P":
                img = img.convert("RGB")
            elif img.mode == "PA":
                img = img.convert("RGBA")

            color_profile_sus = False
            color_mode_corrected = False
            mode_conversion = {
                ('RGBA', 'GRAY'): 'LA',
                ('RGB',  'GRAY'): 'L',
                ('LA',   'RGB '): 'RGBA',
                ('L',    'RGB '): 'RGB',
                ('I;16', 'RGB '): 'RGB',
                ('RGB',  'CMYK'): 'CMYK'
                
            }
            valid_modes = [
                ('RGBA', 'RGB '),
                ('RGB',  'RGB '),
                ('LA',   'GRAY'),
                ('L',    'GRAY'),
                ('I;16', 'GRAY'),
                ('CMYK', 'CMYK')
            ]

            if (img.mode, profile.profile.xcolor_space) not in valid_modes:
                if (img.mode, profile.profile.xcolor_space) in mode_conversion:
                    img = img.convert(mode_conversion[(img.mode, profile.profile.xcolor_space)])
                    color_mode_corrected = True
                else:
                    print(f"WARNING: {fp} has unhandled color space mismatch: '{profile.profile.xcolor_space}' != '{img.mode}'")
                    color_profile_sus = True

            intent_issue = False
            if intent_fallback and not profile.profile.is_intent_supported(intent, ImageCms.Direction.INPUT):
                intent = _coalesce_intent(ImageCms.getDefaultIntent(profile))
                if not not profile.profile.is_intent_supported(intent, ImageCms.Direction.INPUT):
                    print("Warning: This profile doesn't support any operations!")
                    intent_issue = True
                flags = (intent_flags if intent_flags is not None else _INTENT_FLAGS_FALLBACK).get(intent)
            else:
                flags = (intent_flags if intent_flags is not None else _INTENT_FLAGS_INITIAL).get(intent)

            if flags is None:
                raise KeyError(f"no flags for intent {intent}")

            try:
                if img.mode == mode:
                    ImageCms.profileToProfile(
                        img,
                        profile,
                        _SRGB,
                        renderingIntent=intent,
                        inPlace=True,
                        flags=flags
                    )
                else:
                    img = ImageCms.profileToProfile(
                        img,
                        profile,
                        _SRGB,
                        renderingIntent=intent,
                        outputMode=mode,
                        flags=flags
                    )
                if color_profile_sus and not color_mode_corrected:
                    print(f"WARNING: {fp} had a mismatched color profile but loaded fine.")
            except:
                print(f"WARNING: Failed to load color profile for {fp}.  Is it corrupt, or are we mishandling an edge case?")

    if img.mode != mode:
        img = img.convert(mode)

    return img