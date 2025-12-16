#!/usr/bin/env python3
"""
JPG를 ESP32-S3 EPD용 BIN 파일로 변환하는 스크립트

사용법:
    python3 jpg_to_bin.py input.jpg output.bin [--no-dither]
    
옵션:
    --no-dither    디더링 없이 변환 (더 선명하지만 색상 전환이 거칠 수 있음)
"""

import sys
import numpy as np
from PIL import Image
import argparse

# EPD 6색 팔레트 정의 (제조사 제공 PALETTE_AIO_E6)
# 각 색상의 RGB 값 (ok-dev03a 보드용)
EPD_PALETTE = {
    0: (0x00, 0x00, 0x00),    # 검정 (black)
    1: (0xFF, 0xFF, 0xFF),    # 흰색 (white)
    2: (0xFF, 0xFF, 0x00),    # 노랑 (yellow)
    3: (0xFF, 0x00, 0x00),    # 빨강 (red)
    4: (0x00, 0x00, 0xFF),    # 파랑 (blue)
    5: (0x00, 0xFF, 0x00),    # 초록 (green)
}

# 대상 이미지 크기
TARGET_WIDTH = 1200
TARGET_HEIGHT = 1600

# E Ink Spectra6용 8x8 Ordered Dithering 매트릭스 (Bayer)
# C 코드의 matrix_M6와 동일
ORDERED_DITHER_MATRIX = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21]
], dtype=np.uint8)

# Spectra 6 Vivid 파이프라인 기본 파라미터
CHROMA_GAIN = 1.30
CHROMA_THRESHOLD = 0.045
GRAY_SPLIT_L = 0.55
HUE_SNAP_STRENGTH = 0.85
WEIGHT_L = 0.50
WEIGHT_C = 1.20
WEIGHT_H = 1.00
BLUE_NOISE_SIZE = 128
POST_GAMMA = 0.88


# -----------------------------
# OKLab conversion (sRGB <-> OKLab)
# -----------------------------
def srgb_to_linear(srgb):
    """sRGB to linear RGB"""
    a = 0.055
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a)) ** 2.4)

def linear_to_srgb(lin):
    """Linear RGB to sRGB"""
    a = 0.055
    lin = np.clip(lin, 0.0, 1.0)  # 음수 값 방지
    return np.where(lin <= 0.0031308, 12.92 * lin, (1 + a) * np.power(lin, 1/2.4) - a)

def rgb_to_oklab(rgb01):
    """RGB [0,1] to OKLab"""
    lin = srgb_to_linear(rgb01)
    
    # linear sRGB -> LMS (M1)
    M1 = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ], dtype=np.float32)
    lms = lin @ M1.T
    
    lms_cbrt = np.cbrt(np.clip(lms, 1e-12, None))
    
    # LMS' -> OKLab (M2)
    M2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ], dtype=np.float32)
    return lms_cbrt @ M2.T  # (...,3) = [L, a, b]

def oklab_to_rgb(lab):
    """OKLab to RGB [0,1]"""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    # OKLab -> LMS'
    M2_inv = np.array([
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480]
    ], dtype=np.float32)
    lms_ = np.stack([L, a, b], axis=-1) @ M2_inv.T
    
    lms = np.power(lms_, 3.0)
    
    # LMS -> linear sRGB
    M1_inv = np.array([
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010]
    ], dtype=np.float32)
    lin = lms @ M1_inv.T
    
    rgb = linear_to_srgb(lin)
    return np.clip(rgb, 0.0, 1.0)

# -----------------------------
# Blue-noise threshold tile generator
# -----------------------------
def make_blue_noise_tile(n=128, seed=0):
    """Generate blue-noise-like threshold tile"""
    rng = np.random.default_rng(seed)
    x = rng.random((n, n)).astype(np.float32)
    
    # high-pass 느낌 만들기: blur 흉내 (box blur 여러 번)
    def box_blur(a):
        # 3x3 box blur, wrap to make tileable-ish
        s = (
            np.roll(a,  1, 0) + np.roll(a, -1, 0) +
            np.roll(a,  1, 1) + np.roll(a, -1, 1) +
            a +
            np.roll(np.roll(a,  1, 0),  1, 1) +
            np.roll(np.roll(a,  1, 0), -1, 1) +
            np.roll(np.roll(a, -1, 0),  1, 1) +
            np.roll(np.roll(a, -1, 0), -1, 1)
        ) / 9.0
        return s
    
    blur = x.copy()
    for _ in range(6):
        blur = box_blur(blur)
    
    hp = x - blur
    # rank normalize to [0,1] thresholds
    flat = hp.flatten()
    order = np.argsort(flat)
    thresh = np.empty_like(flat)
    thresh[order] = np.linspace(0, 1, flat.size, endpoint=False)
    return thresh.reshape(n, n).astype(np.float32)

# -----------------------------
# Palette in OKLab space
# -----------------------------
def get_palette_oklab():
    """Get palette colors in OKLab space"""
    names = list(EPD_PALETTE.keys())
    rgb = np.array([EPD_PALETTE[k] for k in names], dtype=np.float32) / 255.0
    lab = rgb_to_oklab(rgb)
    return names, rgb, lab

def rgb_to_nearest_color(rgb):
    """RGB 값을 가장 가까운 EPD 팔레트 색상 인덱스로 변환 (기존 방식, 하위 호환)"""
    r, g, b = rgb
    min_distance = float('inf')
    nearest_index = 0
    
    for index, (pr, pg, pb) in EPD_PALETTE.items():
        # 유클리드 거리 계산
        distance = ((r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            nearest_index = index
    
    return nearest_index, EPD_PALETTE[nearest_index]


def eink_default_dither(img_array):
    """
    E Ink Spectra6 기본 디더링 알고리즘 (E6_dithering_sample.c 기반)
    
    C 코드의 dither_image 함수를 정확히 구현:
    1. 경계 제외하고 디더링 (y=1부터 height-1, x=1부터 width-1)
    2. 각 채널(B, G, R)에 대해 값/4 >= 매트릭스값이면 255, 아니면 0
    3. 색상 보정 적용 (magenta, yellow 처리)
    4. 디더링된 결과를 6색 팔레트로 매핑
    """
    height, width = img_array.shape[:2]
    
    # C 코드는 BMP를 읽으므로 BGR 형식입니다
    # PIL은 RGB이므로 BGR로 변환 (B, G, R 순서)
    bgr_array = img_array.copy().astype(np.uint8)
    # RGB -> BGR 변환
    bgr_array[:, :, [0, 2]] = bgr_array[:, :, [2, 0]]
    
    # C 코드: for (int y = 1; y < height - 1; y++)
    # C 코드: for (int x = 1; x < width - 1; x++)
    # 경계 제외하고 디더링 처리
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            matrix_val = ORDERED_DITHER_MATRIX[y % 8, x % 8]
            
            # C 코드: int idx = y * row_padded + x * 3;
            # C 코드: for (int c = 0; c < 3; c++)
            # C 코드: if (data[idx + c] / 4 >= matrix_M6[y % 8][x % 8])
            # 각 채널(B, G, R)에 디더링 적용
            for c in range(3):
                # C의 정수 나눗셈 / 4는 Python의 // 4와 동일
                if (bgr_array[y, x, c] // 4) >= matrix_val:
                    bgr_array[y, x, c] = 255
                else:
                    bgr_array[y, x, c] = 0
    
    # C 코드: 색상 보정 (전체 이미지에 대해, 경계 포함)
    for y in range(height):
        for x in range(width):
            # BGR 형식: data[idx] = B, data[idx+1] = G, data[idx+2] = R
            b = bgr_array[y, x, 0]
            g = bgr_array[y, x, 1]
            r = bgr_array[y, x, 2]
            matrix_val = ORDERED_DITHER_MATRIX[y % 8, x % 8]
            
            # C 코드: if (data[idx] == 255 && data[idx + 1] == 0 && data[idx + 2] == 255)
            # Magenta (B=255, G=0, R=255) 처리
            if b == 255 and g == 0 and r == 255:
                if matrix_val > 32:
                    bgr_array[y, x, 0] = 0
                    bgr_array[y, x, 1] = 0
                    bgr_array[y, x, 2] = 255  # Blue
                else:
                    bgr_array[y, x, 0] = 255
                    bgr_array[y, x, 1] = 0
                    bgr_array[y, x, 2] = 0  # Red
            
            # C 코드: else if (data[idx] == 255 && data[idx + 1] == 255 && data[idx + 2] == 0)
            # Yellow (B=255, G=255, R=0) 처리
            elif b == 255 and g == 255 and r == 0:
                if matrix_val > 32:
                    bgr_array[y, x, 0] = 0
                    bgr_array[y, x, 1] = 255
                    bgr_array[y, x, 2] = 0  # Green
                else:
                    bgr_array[y, x, 0] = 255
                    bgr_array[y, x, 1] = 0
                    bgr_array[y, x, 2] = 0  # Red
    
    # BGR -> RGB 변환
    bgr_array[:, :, [0, 2]] = bgr_array[:, :, [2, 0]]
    
    # 디더링된 결과를 6색 팔레트로 매핑
    result = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            rgb = tuple(bgr_array[y, x])
            nearest_idx, _ = rgb_to_nearest_color(rgb)
            result[y, x] = nearest_idx
    
    return result

def ordered_dither(img_array):
    """E Ink Spectra6용 Ordered Dithering (레거시, eink_default_dither 사용 권장)"""
    return eink_default_dither(img_array)
    
    # C 코드: 색상 보정 (전체 이미지에 대해, 경계 포함)
    # C 코드: for (int y = 0; y < height; y++)
    # C 코드: for (int x = 0; x < width; x++)
    for y in range(height):
        for x in range(width):
            matrix_val = ORDERED_DITHER_MATRIX[y % 8, x % 8]
            
            # C 코드: int idx = y * row_padded + x * 3;
            # C 코드: if (data[idx] == 255 && data[idx + 1] == 0 && data[idx + 2] == 255)
            # B=255, G=0, R=255 -> 마젠타
            if (bgr_array[y, x, 0] == 255 and bgr_array[y, x, 1] == 0 and 
                bgr_array[y, x, 2] == 255):
                if matrix_val > 32:
                    # C 코드: data[idx] = 0; data[idx + 1] = 0; data[idx + 2] = 255; // 藍色 (파랑)
                    # BGR 형식에서: B=0, G=0, R=255 -> RGB로 변환하면 R=255, G=0, B=0 (빨강)
                    bgr_array[y, x, 0] = 0    # B
                    bgr_array[y, x, 1] = 0     # G
                    bgr_array[y, x, 2] = 255   # R
                else:
                    # C 코드: data[idx] = 255; data[idx + 1] = 0; data[idx + 2] = 0; // 紅色 (빨강)
                    # BGR 형식에서: B=255, G=0, R=0 -> RGB로 변환하면 R=0, G=0, B=255 (파랑)
                    bgr_array[y, x, 0] = 255   # B
                    bgr_array[y, x, 1] = 0     # G
                    bgr_array[y, x, 2] = 0     # R
            
            # C 코드: else if (data[idx] == 255 && data[idx + 1] == 255 && data[idx + 2] == 0)
            # B=255, G=255, R=0 -> 시안
            elif (bgr_array[y, x, 0] == 255 and bgr_array[y, x, 1] == 255 and 
                  bgr_array[y, x, 2] == 0):
                if matrix_val > 32:
                    # C 코드: data[idx] = 0; data[idx + 1] = 255; data[idx + 2] = 0; // 綠色 (초록)
                    # BGR 형식에서: B=0, G=255, R=0 -> RGB로 변환하면 R=0, G=255, B=0 (초록)
                    bgr_array[y, x, 0] = 0     # B
                    bgr_array[y, x, 1] = 255   # G
                    bgr_array[y, x, 2] = 0     # R
                else:
                    # C 코드: data[idx] = 255; data[idx + 1] = 0; data[idx + 2] = 0; // 紅色 (빨강)
                    # BGR 형식에서: B=255, G=0, R=0 -> RGB로 변환하면 R=0, G=0, B=255 (파랑)
                    bgr_array[y, x, 0] = 255   # B
                    bgr_array[y, x, 1] = 0     # G
                    bgr_array[y, x, 2] = 0     # R
    
    # BGR을 다시 RGB로 변환
    rgb_array = bgr_array.copy()
    rgb_array[:, :, [0, 2]] = rgb_array[:, :, [2, 0]]
    
    # 최종 결과를 6색 팔레트 인덱스로 변환
    result = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r, g, b = rgb_array[y, x]
            nearest_idx, _ = rgb_to_nearest_color((r, g, b))
            result[y, x] = nearest_idx
    
    return result


def floyd_steinberg_dither(img_array):
    """Floyd-Steinberg 디더링 알고리즘을 사용하여 이미지 변환"""
    height, width = img_array.shape[:2]
    result = np.zeros((height, width), dtype=np.uint8)
    
    # float 배열로 복사 (오차 계산용)
    working = img_array.astype(np.float32)
    
    for y in range(height):
        for x in range(width):
            # 현재 픽셀의 RGB 값
            old_r = np.clip(working[y, x, 0], 0, 255)
            old_g = np.clip(working[y, x, 1], 0, 255)
            old_b = np.clip(working[y, x, 2], 0, 255)
            
            # 가장 가까운 팔레트 색상 찾기
            nearest_idx, (new_r, new_g, new_b) = rgb_to_nearest_color((int(old_r), int(old_g), int(old_b)))
            
            # 결과에 저장
            result[y, x] = nearest_idx
            
            # 오차 계산
            error_r = old_r - new_r
            error_g = old_g - new_g
            error_b = old_b - new_b
            
            # 오차를 주변 픽셀에 분산 (Floyd-Steinberg)
            if x + 1 < width:
                working[y, x + 1] += np.array([error_r * 7/16, error_g * 7/16, error_b * 7/16])
            
            if y + 1 < height:
                if x > 0:
                    working[y + 1, x - 1] += np.array([error_r * 3/16, error_g * 3/16, error_b * 3/16])
                
                working[y + 1, x] += np.array([error_r * 5/16, error_g * 5/16, error_b * 5/16])
                
                if x + 1 < width:
                    working[y + 1, x + 1] += np.array([error_r * 1/16, error_g * 1/16, error_b * 1/16])
    
    return result


def nearest_color_quantize(img_array):
    """디더링 없이 가장 가까운 색상으로 양자화"""
    height, width = img_array.shape[:2]
    result = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            r, g, b = img_array[y, x]
            nearest_idx, _ = rgb_to_nearest_color((r, g, b))
            result[y, x] = nearest_idx
    
    return result


def spectra6_vivid_dither(
    img_array,
    chroma_gain=CHROMA_GAIN,
    chroma_threshold=CHROMA_THRESHOLD,
    gray_split_L=GRAY_SPLIT_L,
    hue_snap_strength=HUE_SNAP_STRENGTH,
    wL=WEIGHT_L,
    wC=WEIGHT_C,
    wH=WEIGHT_H,
    bn_size=BLUE_NOISE_SIZE,
    post_gamma=POST_GAMMA,
    bn_seed=0,
):
    """
    Spectra 6 Vivid Dithering Pipeline
    
    OKLab 기반 색 공간 변환, chroma thresholding, hue snapping,
    palette-aware mapping, blue-noise ordered dithering을 사용하여
    E-ink에서 vivid하게 보이도록 최적화된 변환
    
    Args:
        img_array: HxWx3 uint8 RGB 이미지
        chroma_gain: 채도 증폭 계수 (1.25~1.50)
        chroma_threshold: 채도 임계값, 이보다 낮으면 흑/백으로 snap (0.03~0.06)
        gray_split_L: grayscale snap 시 밝기 기준 (0.50~0.60)
        hue_snap_strength: 색상 방향 고정 강도 (0.70~0.95)
        wL, wC, wH: OKLab 거리 가중치 (밝기, 채도, 색상)
        bn_size: blue-noise tile 크기
        post_gamma: 후처리 감마 값 (0.85~0.90)
        bn_seed: blue-noise 생성 시드
    
    Returns:
        HxW uint8 배열 (팔레트 인덱스 0-5)
    """
    H, W = img_array.shape[:2]
    rgb01 = img_array.astype(np.float32) / 255.0
    lab = rgb_to_oklab(rgb01)
    
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a*a + b*b)
    
    # 1) Chroma exaggeration
    chroma2 = chroma * chroma_gain
    scale = np.where(chroma > 1e-8, chroma2 / chroma, 0.0)
    a2 = a * scale
    b2 = b * scale
    
    # 2) Hue snapping (toward nearest palette hue among chromatic colors)
    names, pal_rgb01, pal_lab = get_palette_oklab()
    # chromatic palette indices (exclude K=0, W=1)
    chromatic_idx = [2, 3, 4, 5]  # Yellow, Red, Blue, Green
    pal_ab = pal_lab[chromatic_idx, 1:3]  # (4,2)
    pal_ab_norm = pal_ab / np.clip(np.linalg.norm(pal_ab, axis=1, keepdims=True), 1e-8, None)
    
    ab = np.stack([a2, b2], axis=-1)  # HxWx2
    ab_norm = ab / np.clip(np.linalg.norm(ab, axis=-1, keepdims=True), 1e-8, None)
    
    # choose nearest hue direction by max dot
    dots = np.tensordot(ab_norm, pal_ab_norm.T, axes=([2],[0]))  # HxWx4
    best = np.argmax(dots, axis=-1)  # HxW
    target_dir = pal_ab_norm[best]   # HxWx2
    
    # blend direction
    new_dir = (1 - hue_snap_strength) * ab_norm + hue_snap_strength * target_dir
    new_dir = new_dir / np.clip(np.linalg.norm(new_dir, axis=-1, keepdims=True), 1e-8, None)
    
    a3 = new_dir[..., 0] * chroma2
    b3 = new_dir[..., 1] * chroma2
    
    lab2 = np.stack([L, a3, b3], axis=-1)
    
    # 3) Chroma threshold -> grayscale snap
    # 주의: chroma가 매우 낮을 때만 grayscale로 snap
    # chroma가 낮아도 약간의 색상 정보가 있으면 유지
    lowc = chroma < chroma_threshold
    labK = pal_lab[0]  # Black
    labW = pal_lab[1]  # White
    # grayscale snap은 매우 낮은 chroma일 때만 적용
    lab2[lowc & (L < gray_split_L)] = labK
    lab2[lowc & (L >= gray_split_L)] = labW
    
    # 4) Palette-aware mapping + 2-color mixing via ordered dithering
    palL = pal_lab[:, 0][None, None, :]  # 1x1x6
    pala = pal_lab[:, 1][None, None, :]
    palb = pal_lab[:, 2][None, None, :]
    
    Lt = lab2[..., 0][..., None]
    at = lab2[..., 1][..., None]
    bt = lab2[..., 2][..., None]
    
    dL = Lt - palL
    da = at - pala
    db = bt - palb
    
    Ct = np.sqrt(lab2[..., 1]**2 + lab2[..., 2]**2)[..., None]
    Cp = np.sqrt((pal_lab[:,1]**2 + pal_lab[:,2]**2))[None, None, :]
    
    # hue distance approx: 1 - cos(angle)
    abt2 = np.stack([lab2[...,1], lab2[...,2]], axis=-1)  # HxWx2
    abp2 = pal_lab[:,1:3]  # 6x2
    abt_norm = abt2 / np.clip(np.linalg.norm(abt2, axis=-1, keepdims=True), 1e-8, None)  # HxWx2
    abp_norm = abp2 / np.clip(np.linalg.norm(abp2, axis=-1, keepdims=True), 1e-8, None)  # 6x2
    cos = np.tensordot(abt_norm, abp_norm.T, axes=([2],[0]))  # HxWx6
    hue_dist = 1.0 - cos  # HxWx6
    
    dist = (wL*(dL**2) + wC*((Ct-Cp)**2) + wH*(hue_dist**2)).astype(np.float32)  # HxWx6
    
    # best and second best
    best_idx = np.argmin(dist, axis=-1)
    dist2 = dist.copy()
    dist2[np.arange(H)[:,None], np.arange(W)[None,:], best_idx] = np.inf
    second_idx = np.argmin(dist2, axis=-1)
    
    c1 = pal_lab[best_idx]     # HxWx3
    c2 = pal_lab[second_idx]   # HxWx3
    t_lab = lab2               # HxWx3
    
    # mixing coefficient t (proportion of c2) by projection onto segment c1->c2
    v = (c2 - c1)
    vv = np.sum(v*v, axis=-1, keepdims=True)
    t = np.sum((t_lab - c1) * v, axis=-1, keepdims=True) / np.clip(vv, 1e-8, None)
    t = np.clip(t, 0.0, 1.0)[..., 0]  # HxW
    
    # blue-noise ordered dithering decision
    bn = make_blue_noise_tile(bn_size, seed=bn_seed)
    u = bn[np.arange(H)[:,None] % bn_size, np.arange(W)[None,:] % bn_size]  # HxW in [0,1)
    use_c2 = (u < t)  # if threshold smaller than proportion -> choose c2
    
    out_lab = np.where(use_c2[..., None], c2, c1)
    
    # 5) Convert back to RGB and apply post-gamma
    out_rgb01 = oklab_to_rgb(out_lab)
    out_rgb01 = np.clip(out_rgb01, 0.0, 1.0) ** post_gamma
    out_rgb = (out_rgb01 * 255.0 + 0.5).astype(np.uint8)
    
    # 6) Map OKLab space에서 직접 팔레트 인덱스 선택 (더 정확)
    # OKLab 공간에서 직접 거리 계산
    out_lab_expanded = out_lab[:, :, None, :]  # (H, W, 1, 3)
    pal_lab_expanded = pal_lab[None, None, :, :]  # (1, 1, 6, 3)
    
    # OKLab 거리 계산 (가중치 적용)
    dL = out_lab_expanded[..., 0] - pal_lab_expanded[..., 0]  # (H, W, 6)
    da = out_lab_expanded[..., 1] - pal_lab_expanded[..., 1]
    db = out_lab_expanded[..., 2] - pal_lab_expanded[..., 2]
    
    # chroma와 hue 거리
    out_C = np.sqrt(out_lab[..., 1]**2 + out_lab[..., 2]**2)[..., None]  # (H, W, 1)
    pal_C = np.sqrt(pal_lab[:, 1]**2 + pal_lab[:, 2]**2)[None, None, :]  # (1, 1, 6)
    
    # hue distance (cosine similarity)
    out_ab = np.stack([out_lab[..., 1], out_lab[..., 2]], axis=-1)  # (H, W, 2)
    pal_ab = pal_lab[:, 1:3]  # (6, 2)
    out_ab_norm = out_ab / np.clip(np.linalg.norm(out_ab, axis=-1, keepdims=True), 1e-8, None)
    pal_ab_norm = pal_ab / np.clip(np.linalg.norm(pal_ab, axis=-1, keepdims=True), 1e-8, None)
    cos_hue = np.tensordot(out_ab_norm, pal_ab_norm.T, axes=([2],[0]))  # (H, W, 6)
    hue_dist = 1.0 - cos_hue
    
    # 가중 거리
    distances = (wL * (dL**2) + wC * ((out_C - pal_C)**2) + wH * (hue_dist**2)).astype(np.float32)
    result = np.argmin(distances, axis=-1).astype(np.uint8)  # (H, W)
    
    return result


def convert_jpg_to_bin(input_jpg, output_bin, use_dithering=True):
    """JPG 파일을 EPD BIN 형식으로 변환"""
    print(f"입력 파일 로딩: {input_jpg}")
    print(f"디더링: {'사용' if use_dithering else '사용 안 함'}")
    
    # 이미지 열기
    img = Image.open(input_jpg)
    print(f"원본 이미지 크기: {img.size} ({img.mode} 모드)")
    
    # RGB 모드로 변환
    if img.mode != 'RGB':
        img = img.convert('RGB')
        print(f"RGB 모드로 변환됨")
    
    # 1200x1600으로 리사이즈 (비율 유지하며 fit)
    img.thumbnail((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
    
    # 캔버스 생성 (흰색 배경)
    canvas = Image.new('RGB', (TARGET_WIDTH, TARGET_HEIGHT), (255, 255, 255))
    
    # 이미지를 캔버스 중앙에 배치
    offset_x = (TARGET_WIDTH - img.width) // 2
    offset_y = (TARGET_HEIGHT - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))
    
    print(f"리사이즈 완료: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    
    # numpy 배열로 변환
    img_array = np.array(canvas, dtype=np.uint8)
    
    # E Ink 기본 디더링 사용 (기본값)
    if use_dithering:
        print("E Ink Default Dithering 적용 중...")
        print("  (8x8 Bayer matrix, official E Ink algorithm)")
        data = eink_default_dither(img_array)
    else:
        print("가장 가까운 색상으로 양자화 중...")
        data = nearest_color_quantize(img_array)
    
    print("팔레트 양자화 완료")
    
    # 값 분포 확인
    unique, counts = np.unique(data, return_counts=True)
    print("\n색상 분포:")
    for value, count in zip(unique, counts):
        percentage = (count / data.size) * 100
        print(f"  색상 {value}: {count:7d} 픽셀 ({percentage:5.2f}%)")
    
    # 바이너리 파일로 저장 (헤더 없이 순수 픽셀 데이터만)
    with open(output_bin, 'wb') as f:
        f.write(data.tobytes())
    
    print(f"\n변환 완료: {output_bin}")
    print(f"파일 크기: {data.size} bytes ({TARGET_WIDTH} x {TARGET_HEIGHT})")


def main():
    parser = argparse.ArgumentParser(
        description='JPG를 ESP32-S3 ok-dev03a 보드용 EPD BIN 파일로 변환',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 디더링 사용 (기본값, 부드러운 색상 전환)
  python3 jpg_to_bin.py input.jpg output.bin
  
  # 디더링 없이 변환 (더 선명한 색상)
  python3 jpg_to_bin.py input.jpg output.bin --no-dither
  
색상 팔레트:
  0: 검정 (Black)
  1: 흰색 (White)
  2: 노랑 (Yellow)
  3: 빨강 (Red)
  4: 파랑 (Blue)
  5: 초록 (Green)

기본 알고리즘: Spectra 6 Vivid Dithering
  - OKLab 색 공간 기반 변환
  - Chroma thresholding 및 exaggeration
  - Hue snapping
  - Blue-noise ordered dithering
  - E-ink에서 vivid하게 보이도록 최적화
        """
    )
    
    parser.add_argument('input_jpg', help='입력 JPG 파일 경로')
    parser.add_argument('output_bin', help='출력 BIN 파일 경로')
    parser.add_argument('--no-dither', action='store_true', 
                        help='디더링 비활성화 (기본값: Floyd-Steinberg 디더링 사용)')
    
    args = parser.parse_args()
    
    try:
        convert_jpg_to_bin(args.input_jpg, args.output_bin, use_dithering=not args.no_dither)
    except FileNotFoundError as e:
        print(f"\n오류: 파일을 찾을 수 없습니다 - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

