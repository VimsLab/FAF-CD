"""Test-time pseudo-change perturbations for robustness evaluation.

Applied via ``ValPre`` during evaluation only. Perturbations are deterministic
per-sample (seeded from filename + global seed) so re-runs are reproducible.
"""

import hashlib
from typing import Optional

import cv2
import numpy as np


PERTURBATION_KINDS = (
    "brightness_contrast",
    "color_jitter",
    "gaussian_noise",
    "speckle_noise",
    "radiometric_shift",
    "gaussian_blur",
    "motion_blur",
    "resolution_blur",
    "haze",
    "shadow",
)

APPLY_TO_CHOICES = ("A", "B", "both")


def _check_severity(severity: int) -> int:
    severity = int(severity)
    if not 1 <= severity <= 5:
        raise ValueError(f"perturbation severity must be in [1,5], got {severity}")
    return severity


def _seed_for_sample(global_seed: int, sample_id: str, branch: str) -> int:
    h = hashlib.md5(f"{global_seed}|{sample_id}|{branch}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big", signed=False)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def brightness_contrast(img: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    # severity controls the magnitude of multiplicative contrast and additive brightness;
    # gamma shift is folded in to also cover non-linear illumination changes.
    contrast_range = [0.05, 0.10, 0.18, 0.25, 0.35][severity - 1]
    bright_range = [10, 20, 35, 50, 70][severity - 1]
    gamma_range = [0.10, 0.18, 0.28, 0.40, 0.55][severity - 1]

    contrast = 1.0 + rng.uniform(-contrast_range, contrast_range)
    brightness = rng.uniform(-bright_range, bright_range)
    gamma = 1.0 + rng.uniform(-gamma_range, gamma_range)

    out = img.astype(np.float32) * contrast + brightness
    out = np.clip(out, 0, 255) / 255.0
    out = np.power(out, gamma) * 255.0
    return _to_uint8(out)


def color_jitter(img: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    hue_range = [3, 6, 10, 14, 20][severity - 1]
    sat_range = [0.10, 0.18, 0.28, 0.38, 0.50][severity - 1]
    val_range = [0.05, 0.10, 0.15, 0.22, 0.30][severity - 1]

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue_range, hue_range)) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + rng.uniform(-sat_range, sat_range)), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + rng.uniform(-val_range, val_range)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def gaussian_noise(img: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    sigma = [5, 10, 18, 26, 38][severity - 1]
    noise = rng.randn(*img.shape).astype(np.float32) * sigma
    return _to_uint8(img.astype(np.float32) + noise)


def speckle_noise(img: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    sigma = [0.06, 0.10, 0.16, 0.24, 0.34][severity - 1]
    noise = rng.randn(*img.shape).astype(np.float32) * sigma
    out = img.astype(np.float32) * (1.0 + noise)
    return _to_uint8(out)


def radiometric_shift(img: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    gain_range = [0.06, 0.10, 0.16, 0.24, 0.34][severity - 1]
    offset_range = [6, 12, 22, 34, 48][severity - 1]
    gamma_range = [0.08, 0.14, 0.22, 0.32, 0.45][severity - 1]

    gain = 1.0 + rng.uniform(-gain_range, gain_range)
    offset = rng.uniform(-offset_range, offset_range)
    gamma = 1.0 + rng.uniform(-gamma_range, gamma_range)

    out = img.astype(np.float32) * gain + offset
    out = np.clip(out, 0, 255) / 255.0
    out = np.power(out, gamma) * 255.0
    return _to_uint8(out)


def gaussian_blur(img: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    sigma = [0.6, 1.0, 1.6, 2.4, 3.5][severity - 1]
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def motion_blur(img: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    length = [3, 6, 10, 15, 22][severity - 1]
    angle = float(rng.uniform(0, 180))
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    if img.ndim == 2:
        return cv2.filter2D(img, -1, kernel)
    channels = [cv2.filter2D(img[..., c], -1, kernel) for c in range(img.shape[2])]
    return np.stack(channels, axis=-1)


def resolution_blur(img: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    keep_singleton_channel = img.ndim == 3 and img.shape[2] == 1
    scale = [0.85, 0.72, 0.58, 0.45, 0.34][severity - 1]
    sigma = [0.2, 0.35, 0.55, 0.8, 1.1][severity - 1]
    h, w = img.shape[:2]
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)
    if keep_singleton_channel and out.ndim == 2:
        out = out[:, :, np.newaxis]
    return _to_uint8(out)


def haze(img: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    # Atmospheric scattering model: I = J * t + A * (1 - t).
    # Spatially varying transmission t simulates thin cloud / haze without fully
    # occluding ground content (severity caps the maximum opacity at < 1).
    max_alpha = [0.12, 0.22, 0.34, 0.46, 0.58][severity - 1]
    h, w = img.shape[:2]

    coarse_h, coarse_w = max(8, h // 32), max(8, w // 32)
    coarse = rng.rand(coarse_h, coarse_w).astype(np.float32)
    coarse = cv2.GaussianBlur(coarse, (0, 0), sigmaX=2.0)
    cmin, cmax = float(coarse.min()), float(coarse.max())
    if cmax > cmin:
        coarse = (coarse - cmin) / (cmax - cmin)
    alpha = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR) * max_alpha

    airlight = float(rng.uniform(210, 250))
    if img.ndim == 3:
        alpha = alpha[..., None]
    out = img.astype(np.float32) * (1.0 - alpha) + airlight * alpha
    return _to_uint8(out)


def shadow(img: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    # Build a soft-edged dark polygon mask covering 5-40% of the image,
    # then attenuate intensity inside it. Mimics cast shadows from buildings
    # / clouds, which are a common pseudo-change source in remote sensing.
    darkness = [0.15, 0.28, 0.42, 0.55, 0.68][severity - 1]
    coverage_max = [0.10, 0.18, 0.28, 0.38, 0.48][severity - 1]
    h, w = img.shape[:2]

    cx = int(rng.uniform(0, w))
    cy = int(rng.uniform(0, h))
    target_area = h * w * rng.uniform(0.05, coverage_max)
    radius = max(8, int(np.sqrt(target_area / np.pi) * rng.uniform(0.8, 1.4)))
    n_pts = int(rng.randint(5, 9))
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=n_pts))
    radii = radius * rng.uniform(0.6, 1.3, size=n_pts)
    pts = np.stack(
        [cx + radii * np.cos(angles), cy + radii * np.sin(angles)],
        axis=-1,
    ).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [pts], 1.0)
    blur_k = max(7, (max(h, w) // 32) | 1)
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    mask = np.clip(mask, 0.0, 1.0)

    attenuation = 1.0 - mask * darkness
    if img.ndim == 3:
        attenuation = attenuation[..., None]
    return _to_uint8(img.astype(np.float32) * attenuation)


_KIND_TO_FN = {
    "brightness_contrast": brightness_contrast,
    "color_jitter": color_jitter,
    "gaussian_noise": gaussian_noise,
    "speckle_noise": speckle_noise,
    "radiometric_shift": radiometric_shift,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "resolution_blur": resolution_blur,
    "haze": haze,
    "shadow": shadow,
}


class TestPerturbation:
    """Apply a single perturbation kind to A, B, or both branches.

    ``apply_to='both'`` perturbs each branch with an independent random draw so
    the simulated cross-time appearance shift is realistic.
    """

    def __init__(self, kind: str, severity: int, apply_to: str = "both",
                 seed: int = 0):
        if kind not in _KIND_TO_FN:
            raise ValueError(
                f"Unknown perturbation kind '{kind}'. "
                f"Choices: {sorted(_KIND_TO_FN.keys())}"
            )
        if apply_to not in APPLY_TO_CHOICES:
            raise ValueError(
                f"apply_to must be one of {APPLY_TO_CHOICES}, got '{apply_to}'"
            )
        self.kind = kind
        self.severity = _check_severity(severity)
        self.apply_to = apply_to
        self.seed = int(seed)
        self._fn = _KIND_TO_FN[kind]
        self._sample_id: Optional[str] = None

    def set_sample_id(self, sample_id: str) -> None:
        self._sample_id = str(sample_id)

    def _apply(self, img: np.ndarray, branch: str) -> np.ndarray:
        sample_id = self._sample_id if self._sample_id is not None else ""
        rng = np.random.RandomState(_seed_for_sample(self.seed, sample_id, branch))
        return self._fn(img, self.severity, rng)

    def __call__(self, A: np.ndarray, B: np.ndarray):
        if self.apply_to in ("A", "both"):
            A = self._apply(A, "A")
        if self.apply_to in ("B", "both"):
            B = self._apply(B, "B")
        return A, B
