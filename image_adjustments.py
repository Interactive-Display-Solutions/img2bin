#!/usr/bin/env python3
"""
이미지 조정 함수들 (contrast, brightness, saturation, hue)
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import colorsys

def adjust_image(img_array, contrast=1.0, brightness=1.0, saturation=1.0, hue_shift=0.0, smoothing=0.0):
    """
    이미지 조정 (contrast, brightness, saturation, hue, smoothing)
    
    Args:
        img_array: HxWx3 uint8 RGB 이미지
        contrast: 대비 조정 (0.0 ~ 2.0, 1.0 = 원본)
        brightness: 밝기 조정 (0.0 ~ 2.0, 1.0 = 원본)
        saturation: 채도 조정 (0.0 ~ 2.0, 1.0 = 원본)
        hue_shift: 색상 회전 (-180 ~ 180도, 0 = 원본)
        smoothing: 부드러움 조정 (0.0 ~ 2.0, 0.0 = 원본, 높을수록 더 부드러움)
    
    Returns:
        조정된 HxWx3 uint8 RGB 이미지
    """
    # PIL Image로 변환
    img = Image.fromarray(img_array)
    
    # Smoothing 적용 (먼저 적용하여 점 패턴을 부드럽게)
    if smoothing > 0.0:
        # Gaussian blur 사용 (smoothing 값에 따라 강도 조절)
        # smoothing 0.0 = blur 없음, 1.0 = 약간 blur, 2.0 = 강한 blur
        blur_radius = smoothing * 0.5  # 최대 1.0 픽셀 blur
        if blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Contrast 조정
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
    
    # Brightness 조정
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    
    # Saturation 조정
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)
    
    # Hue shift (HSV 공간에서 조정) - 간단한 버전
    if hue_shift != 0.0:
        img_array = np.array(img, dtype=np.float32) / 255.0
        hsv_result = np.zeros_like(img_array)
        
        # 각 픽셀을 처리 (작은 이미지이므로 루프 사용)
        H, W = img_array.shape[:2]
        for y in range(H):
            for x in range(W):
                r, g, b = img_array[y, x]
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                # Hue shift
                h = (h * 360.0 + hue_shift) % 360.0 / 360.0
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                hsv_result[y, x] = [r, g, b]
        
        img_array = (np.clip(hsv_result, 0.0, 1.0) * 255.0).astype(np.uint8)
        img = Image.fromarray(img_array)
    
    return np.array(img, dtype=np.uint8)

