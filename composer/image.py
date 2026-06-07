from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass
from os import PathLike
from typing import Literal, Optional, Union
from urllib import request
from urllib.parse import urlparse

from PIL import Image, ImageEnhance

ImageFormat = Literal["jpeg", "png", "webp"]
VisionDetail = Literal["low", "high", "auto"]
ImageSource = Union[str, PathLike[str]]

MIME_BY_FORMAT: dict[ImageFormat, str] = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


@dataclass
class ImageAttach:
    """Image preprocessing and vision API options for attach / ImageMessage."""

    max_width: Optional[int] = None
    max_height: Optional[int] = None
    max_pixels: Optional[int] = None
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    format: ImageFormat = "jpeg"
    quality: int = 85
    detail: VisionDetail = "auto"
    process_url: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.quality <= 100:
            raise ValueError("quality must be between 1 and 100")
        if self.brightness < 0:
            raise ValueError("brightness must be non-negative")
        if self.contrast < 0:
            raise ValueError("contrast must be non-negative")
        if self.saturation < 0:
            raise ValueError("saturation must be non-negative")

    def needs_preprocessing(self) -> bool:
        return (
            self.max_width is not None
            or self.max_height is not None
            or self.max_pixels is not None
            or self.brightness != 1.0
            or self.contrast != 1.0
            or self.saturation != 1.0
        )


def resolve_config(
    config: ImageAttach | None = None, **kwargs: object
) -> ImageAttach:
    if config is not None and kwargs:
        raise ValueError("Pass either config or keyword options, not both")
    if config is not None:
        return config
    return ImageAttach(**kwargs)  # type: ignore[arg-type]


def _normalize_source(source: ImageSource) -> str:
    if isinstance(source, str):
        return source
    return os.fspath(source)


def _is_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")


def _fetch_url(url: str) -> bytes:
    try:
        with request.urlopen(url, timeout=30) as resp:
            return resp.read()
    except Exception as exc:
        raise ValueError(f"Failed to fetch image URL {url!r}: {exc}") from exc


def load_image(source: ImageSource) -> Image.Image:
    source = _normalize_source(source)
    if _is_url(source):
        data = _fetch_url(source)
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            return img
        except Exception as exc:
            raise ValueError(
                f"Failed to decode image from URL {source!r}: {exc}"
            ) from exc
    try:
        img = Image.open(source)
        img.load()
        return img
    except FileNotFoundError as exc:
        raise ValueError(f"Image file not found: {source!r}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to open image {source!r}: {exc}") from exc


def _resize_image(img: Image.Image, config: ImageAttach) -> Image.Image:
    width, height = img.size
    scale = 1.0

    if config.max_width is not None and width > config.max_width:
        scale = min(scale, config.max_width / width)
    if config.max_height is not None and height > config.max_height:
        scale = min(scale, config.max_height / height)
    if config.max_pixels is not None:
        pixels = width * height
        if pixels > config.max_pixels:
            scale = min(scale, math.sqrt(config.max_pixels / pixels))

    if scale < 1.0:
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def _apply_color_grade(img: Image.Image, config: ImageAttach) -> Image.Image:
    if config.brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(config.brightness)
    if config.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(config.contrast)
    if config.saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(config.saturation)
    return img


def _prepare_for_encode(img: Image.Image, fmt: ImageFormat) -> Image.Image:
    if fmt in ("jpeg", "webp") and img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            background.paste(img, mask=img.split()[-1])
        else:
            background.paste(img)
        return background
    if img.mode not in ("RGB", "L"):
        return img.convert("RGB")
    return img


def preprocess_image(img: Image.Image, config: ImageAttach) -> bytes:
    img = _resize_image(img, config)
    img = _apply_color_grade(img, config)
    img = _prepare_for_encode(img, config.format)

    buffer = io.BytesIO()
    save_kwargs: dict = {}
    if config.format in ("jpeg", "webp"):
        save_kwargs["quality"] = config.quality

    fmt_upper = config.format.upper()
    if fmt_upper == "JPEG":
        save_kwargs.setdefault("optimize", True)
    img.save(buffer, format=fmt_upper, **save_kwargs)
    return buffer.getvalue()


def to_data_url(data: bytes, mime: str) -> str:
    encoded = base64.standard_b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _should_passthrough_url(source: str, config: ImageAttach) -> bool:
    if not _is_url(source):
        return False
    if config.process_url:
        return False
    return not config.needs_preprocessing()


def _url_for_source(source: ImageSource, config: ImageAttach) -> str:
    source = _normalize_source(source)
    if _should_passthrough_url(source, config):
        return source
    img = load_image(source)
    data = preprocess_image(img, config)
    mime = MIME_BY_FORMAT[config.format]
    return to_data_url(data, mime)


def build_image_block(
    source: ImageSource,
    config: ImageAttach | None = None,
    **kwargs: object,
) -> dict:
    cfg = resolve_config(config, **kwargs)
    url = _url_for_source(source, cfg)
    image_url: dict = {"url": url, "detail": cfg.detail}
    return {"type": "image_url", "image_url": image_url}


def merge_image_content(existing: object, image_block: dict) -> list:
    if isinstance(existing, str):
        if existing.strip():
            return [{"type": "text", "text": existing}, image_block]
        return [image_block]
    if isinstance(existing, list):
        return list(existing) + [image_block]
    if not existing:
        return [image_block]
    return [{"type": "text", "text": str(existing)}, image_block]
