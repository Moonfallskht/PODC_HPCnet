# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

# ========== 配置 ==========
CONFIG = {
    "input_path":  r"Z:\我的文档\成像\单像素成像\成像图像\数据处理\文章数据处理\不同噪声模拟\GT\分辨率板.bmp",
    "output_dir":  r"Z:\net\ponet0.01\分数据集分",

    # 雾气与噪声参数
    "beta": 15.0,               # 雾气浓度
    "A": 120.0,                # 大气光亮度
    "depth_fixed": 0.2,       # 固定雾气深度（0~1）
    "blur_sigma": 0.5,         # 模糊程度
    "scatter_std": 0.1,        # 高斯噪声强度
    "distort_strength": 0.2,   # 湍流畸变强度
    "illum_strength": 0.8,     # 光照不均强度
    "illum_mode": "radial",    # 光照模式
    "shot_strength": 0.1       # 散粒噪声强度
}

# ========== 函数 ==========
def add_shot_noise(img_float_255: np.ndarray, strength: float = 0.3) -> np.ndarray:
    scale = 1.0 / max(strength, 1e-6)
    photons = np.clip(img_float_255 * scale, 0, None)
    noisy_photons = np.random.poisson(photons).astype(np.float32)
    noisy_img = noisy_photons / scale
    return np.clip(noisy_img, 0, 255)

# ========== 主程序 ==========
def main():
    cfg = CONFIG
    input_path = Path(cfg["input_path"])
    output_dir = Path(cfg["output_dir"])

    gt_dir = output_dir / "GT"
    noisy_dir = output_dir / "noisy"
    gt_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    # 读取 GT 图像
    gt_img = Image.open(input_path).convert("L")
    gt_np = np.array(gt_img, dtype=np.float32)

    H, W = gt_np.shape
    depth = cfg["depth_fixed"]
    t = np.exp(-cfg["beta"] * depth)
    haze_img = gt_np * t + cfg["A"] * (1 - t)

    # 光照不均
    if cfg["illum_strength"] > 0:
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = np.random.randint(0, W), np.random.randint(0, H)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        dist /= dist.max() + 1e-6
        alpha = 3.0
        illum = np.exp(-alpha * dist)
        illum = 1 - cfg["illum_strength"] + cfg["illum_strength"] * (illum / (illum.max() + 1e-6))
        haze_img *= illum

    # 模糊
    blurred = gaussian_filter(haze_img, sigma=cfg["blur_sigma"])
    scatter = np.random.normal(0, cfg["scatter_std"], (H, W)).astype(np.float32)
    turbid_img = np.clip(blurred + scatter, 0, 255)

    # 湍流畸变
    if cfg["distort_strength"] > 0:
        dx = (np.random.rand(H, W).astype(np.float32) - 0.5) * cfg["distort_strength"]
        dy = (np.random.rand(H, W).astype(np.float32) - 0.5) * cfg["distort_strength"]
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        coords = np.array([
            np.clip(y + dy, 0, H - 1),
            np.clip(x + dx, 0, W - 1)
        ])
        distorted = map_coordinates(turbid_img, coords, order=1, mode='reflect')
    else:
        distorted = turbid_img

    # 散粒噪声
    noisy_img = add_shot_noise(distorted, strength=cfg["shot_strength"])

    gt_name = input_path.stem
    gt_save = gt_dir / f"{gt_name}.bmp"
    noisy_subdir = noisy_dir / gt_name
    noisy_subdir.mkdir(parents=True, exist_ok=True)
    noisy_save = noisy_subdir / f"{gt_name}_noisy.bmp"

    gt_img.save(gt_save)
    Image.fromarray(noisy_img.astype(np.uint8)).save(noisy_save)

    print(f"生成噪声图：\n  GT: {gt_save}\n  Noisy: {noisy_save}")

if __name__ == "__main__":
    main()
