import os
import math
from typing import Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve


def load_image_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert('L')
    return np.asarray(img, dtype=np.float32) / 255.0


def save_image_gray(path: str, img: np.ndarray) -> None:
    img_clip = np.clip(img, 0.0, 1.0)
    Image.fromarray((img_clip * 255.0).astype(np.uint8)).save(path)


def gaussian_kernel(sigma: float, size: int) -> np.ndarray:
    assert size % 2 == 1, "Kernel size must be odd"
    radius = size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return fftconvolve(image, kernel, mode='same')


def estimate_gaussian_kernel_params(
    gt: np.ndarray,
    blurred: np.ndarray,
    sigma_range=(0.5, 3.0),
    num_sigma=10,
    size_options=(3, 5, 7, 9, 11),
) -> Tuple[float, int, float]:
    """
    Grid search sigma and size to minimize MSE between conv(gt, k) + b and blurred.
    Returns (best_sigma, best_size, best_bias_b)
    """
    best_sigma, best_size, best_b = 1.5, 5, 0.0
    best_loss = float('inf')
    sigmas = np.linspace(sigma_range[0], sigma_range[1], num_sigma, dtype=np.float32)
    for size in size_options:
        for sigma in sigmas:
            k = gaussian_kernel(float(sigma), int(size))
            pred = convolve_same(gt, k)
            b = float(np.mean(blurred - pred))
            diff = (pred + b) - blurred
            loss = float(np.mean(diff**2))
            if loss < best_loss:
                best_loss = loss
                best_sigma, best_size, best_b = float(sigma), int(size), float(b)
    return best_sigma, best_size, best_b


def laplacian_kernel() -> np.ndarray:
    # 2D 4-neighbour Laplacian
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def save_kernel_heatmap(path: str, kernel: np.ndarray, upscale: int = 24) -> None:
    """
    Save a heatmap-like visualization of the kernel by upscaling the kernel with nearest-neighbor.
    """
    k = kernel.astype(np.float32)
    k = (k - k.min()) / (k.max() - k.min() + 1e-12)
    # Nearest-neighbor upscale via Kronecker product
    up = np.kron(k, np.ones((upscale, upscale), dtype=np.float32))
    img = (up * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_blur_effect_grid(path: str, gt: np.ndarray, kernel: np.ndarray, bias_b: float) -> None:
    """
    Save a side-by-side visualization: [GT | Convolved+Bias | |Diff| heatmap].
    """
    pred = convolve_same(gt, kernel) + float(bias_b)
    pred = np.clip(pred, 0.0, 1.0)
    diff = np.abs(pred - gt)
    # Normalize diff for display
    d = diff / (diff.max() + 1e-12)

    def to_u8(x: np.ndarray) -> np.ndarray:
        return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)

    gt_u8 = to_u8(gt)
    pred_u8 = to_u8(pred)
    diff_u8 = to_u8(d)

    h, w = gt_u8.shape
    canvas = np.zeros((h, w * 3), dtype=np.uint8)
    canvas[:, 0:w] = gt_u8
    canvas[:, w:2*w] = pred_u8
    canvas[:, 2*w:3*w] = diff_u8
    Image.fromarray(canvas).save(path)


def biharmonic(image: np.ndarray) -> np.ndarray:
    # Δ(Δx) via two Laplacian convolutions
    lap = convolve_same(image, laplacian_kernel())
    return convolve_same(lap, laplacian_kernel())


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def fista_sparse_sim(
    f: np.ndarray,
    kernel: np.ndarray,
    bias_b: float,
    lambda_data: float,
    lambda_l1: float,
    hessian_weight: float,
    num_iters: int = 200,
    step_size: Optional[float] = None,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Minimize: (lambda/2)||f - A x - b||^2 + hessian_weight * ||Δx||^2 + lambda_l1 ||x||_1
    A is convolution with kernel; A^T uses flipped kernel.
    Uses FISTA for L1, with smooth term gradient combining data and biharmonic regularizer.
    """
    if x0 is None:
        x = f.copy()
    else:
        x = x0.copy()
    y = x.copy()
    t = 1.0

    # Precompute flipped kernel for A^T
    kflip = np.flipud(np.fliplr(kernel))

    # Estimate Lipschitz constant L approximately via power iteration on A^T A and Δ^2
    if step_size is None:
        probe = np.random.randn(*f.shape).astype(np.float32)
        probe /= np.linalg.norm(probe) + 1e-12
        for _ in range(10):
            Ap = convolve_same(probe, kernel)
            Ata_p = convolve_same(Ap, kflip)
            Bhp = biharmonic(probe)
            probe = lambda_data * Ata_p + 2.0 * hessian_weight * Bhp
            n = np.linalg.norm(probe)
            if n < 1e-12:
                break
            probe /= n
        L = np.linalg.norm(lambda_data * Ata_p + 2.0 * hessian_weight * Bhp) + 1e-6
        step = 1.0 / L
    else:
        step = float(step_size)

    for _ in range(num_iters):
        Ay = convolve_same(y, kernel)
        resid = (Ay + bias_b) - f
        grad_data = convolve_same(resid, kflip)  # A^T(Ay + b - f)
        grad_hess = biharmonic(y)
        grad = lambda_data * grad_data + 2.0 * hessian_weight * grad_hess

        x_prev = x
        x = soft_threshold(y - step * grad, step * lambda_l1)
        x = np.clip(x, 0.0, 1.0)

        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = x + ((t - 1.0) / t_next) * (x - x_prev)
        t = t_next

    return x


def _radial_bins(h: int, w: int, num_rings: int) -> Tuple[np.ndarray, np.ndarray]:
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = np.max(rr)
    # Map radius to [0, num_rings)
    ring_idx = np.clip((rr / (r_max + 1e-12)) * num_rings, 0, num_rings - 1e-6).astype(np.int32)
    # Frequency normalized to Nyquist (0..0.5 cycles/pixel)
    freqs = (np.arange(num_rings, dtype=np.float32) / max(num_rings - 1, 1)) * 0.5
    return ring_idx, freqs


def compute_frc(img1: np.ndarray, img2: np.ndarray, num_rings: int = 64) -> Dict[str, Any]:
    img1f = np.asarray(img1, dtype=np.float32)
    img2f = np.asarray(img2, dtype=np.float32)
    h, w = img1f.shape
    # Remove mean to reduce DC dominance
    a = img1f - float(np.mean(img1f))
    b = img2f - float(np.mean(img2f))
    Fa = np.fft.fftshift(np.fft.fft2(a))
    Fb = np.fft.fftshift(np.fft.fft2(b))
    ring_idx, freqs = _radial_bins(h, w, num_rings)

    num = np.zeros(num_rings, dtype=np.complex64)
    den_a = np.zeros(num_rings, dtype=np.float32)
    den_b = np.zeros(num_rings, dtype=np.float32)
    # Accumulate per ring
    for r in range(num_rings):
        mask = (ring_idx == r)
        Fa_r = Fa[mask]
        Fb_r = Fb[mask]
        if Fa_r.size == 0:
            continue
        num[r] = np.sum(Fa_r * np.conj(Fb_r))
        den_a[r] = float(np.sum(np.abs(Fa_r) ** 2))
        den_b[r] = float(np.sum(np.abs(Fb_r) ** 2))

    denom = np.sqrt(den_a * den_b) + 1e-12
    frc_curve = np.real(num) / denom
    frc_curve = np.clip(frc_curve, -1.0, 1.0)

    # Summaries
    auc = float(np.trapz(np.maximum(frc_curve, 0.0), freqs))
    # Focus on mid-high frequencies (>= 0.25 Nyquist)
    high_mask = freqs >= 0.25
    if np.any(high_mask):
        high_auc = float(np.trapz(np.maximum(frc_curve[high_mask], 0.0), freqs[high_mask]))
    else:
        high_auc = 0.0

    # 1/7 threshold resolution (approx). Find highest freq where FRC > 1/7.
    threshold = 1.0 / 7.0
    above = np.where(frc_curve > threshold)[0]
    halfbit_res = float(freqs[above[-1]]) if above.size > 0 else 0.0

    return {
        'freqs': freqs.tolist(),
        'frc_curve': frc_curve.tolist(),
        'frc_auc': auc,
        'frc_high_auc': high_auc,
        'frc_halfbit_res': halfbit_res,
    }


def compute_rolling_frc(
    img1: np.ndarray,
    img2: np.ndarray,
    tile: int = 64,
    stride: int = 32,
    num_rings: int = 32,
) -> Dict[str, Any]:
    h, w = img1.shape
    curves = []
    for y in range(0, max(h - tile + 1, 1), stride):
        for x in range(0, max(w - tile + 1, 1), stride):
            patch1 = img1[y:y+tile, x:x+tile]
            patch2 = img2[y:y+tile, x:x+tile]
            if patch1.shape != (tile, tile) or patch2.shape != (tile, tile):
                continue
            frc = compute_frc(patch1, patch2, num_rings=num_rings)
            curves.append(np.asarray(frc['frc_curve'], dtype=np.float32))
    if not curves:
        return {
            'rfrc_curve_mean': [],
            'rfrc_curve_std': [],
            'rfrc_auc_mean': 0.0,
            'rfrc_auc_std': 0.0,
        }
    curves_arr = np.stack(curves, axis=0)
    mean_curve = np.mean(curves_arr, axis=0)
    std_curve = np.std(curves_arr, axis=0)
    freqs = (np.arange(num_rings, dtype=np.float32) / max(num_rings - 1, 1)) * 0.5
    auc_vals = np.trapz(np.maximum(curves_arr, 0.0), freqs, axis=1)
    return {
        'rfrc_curve_mean': mean_curve.tolist(),
        'rfrc_curve_std': std_curve.tolist(),
        'rfrc_auc_mean': float(np.mean(auc_vals)),
        'rfrc_auc_std': float(np.std(auc_vals)),
    }


def compute_metrics(restored: np.ndarray, gt: Optional[np.ndarray]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if gt is not None:
        frc = compute_frc(restored, gt, num_rings=64)
        rfrc = compute_rolling_frc(restored, gt, tile=64, stride=32, num_rings=32)
        metrics.update(frc)
        metrics.update(rfrc)
    # 作为补充的锐度代理：梯度均值
    gx = np.gradient(restored, axis=1)
    gy = np.gradient(restored, axis=0)
    metrics['mean_grad_mag'] = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    return metrics


def infer_pair_paths(images_dir: str, pick_blur_name: Optional[str] = None) -> Tuple[str, str]:
    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png') or '.' not in f])
    # 支持新的模糊图命名：Cell_XXX_level_YY(.png 可选)
    level_re = re.compile(r"^Cell_(?P<cell>\d{3})_level_(?P<level>\d{2})(?:\.png)?$", re.IGNORECASE)

    if pick_blur_name is None:
        candidates = [f for f in files if level_re.match(f)]
        if not candidates:
            raise FileNotFoundError('未找到模糊图：应匹配 Cell_XXX_level_YY(.png)')
        pick_blur_name = candidates[0]
    blurred_path = os.path.join(images_dir, pick_blur_name)

    m = level_re.match(pick_blur_name)
    if not m:
        raise FileNotFoundError('指定的模糊图文件名不符合 Cell_XXX_level_YY(.png)')
    cell = m.group('cell')

    # 新的 GT 命名：Cell_XXX_RawSIMData_gt(.png 可选)，优先带 .png
    gt_candidates = [f"Cell_{cell}_RawSIMData_gt.png", f"Cell_{cell}_RawSIMData_gt"]
    gt_name = None
    for cand in gt_candidates:
        if cand in files:
            gt_name = cand
            break
    if gt_name is None:
        # 兜底：在目录中搜索以 Cell_XXX_RawSIMData_gt 开头的文件
        for f in files:
            if f.lower().startswith(f"cell_{cell}_rawsimdata_gt".lower()):
                gt_name = f
                break
    if gt_name is None:
        raise FileNotFoundError(f'未找到GT：期望 Cell_{cell}_RawSIMData_gt(.png)')

    gt_path = os.path.join(images_dir, gt_name)
    return gt_path, blurred_path


__all__ = [
    'load_image_gray',
    'save_image_gray',
    'gaussian_kernel',
    'estimate_gaussian_kernel_params',
    'fista_sparse_sim',
    'compute_metrics',
    'infer_pair_paths',
]


