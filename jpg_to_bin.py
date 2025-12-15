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


def rgb_to_nearest_color(rgb):
    """RGB 값을 가장 가까운 EPD 팔레트 색상 인덱스로 변환"""
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


def ordered_dither(img_array):
    """E Ink Spectra6용 Ordered Dithering (C 파일 로직 정확히 따름)
    
    C 코드의 dither_image 함수를 정확히 구현:
    1. 경계 제외하고 디더링 (y=1부터 height-1, x=1부터 width-1)
    2. 각 채널(B, G, R)에 대해 값/4 >= 매트릭스값이면 255, 아니면 0
    3. 색상 보정 적용
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
    
    # 직접 구현한 색상 양자화 사용 (정확한 6색 팔레트 매칭)
    if use_dithering:
        print("Ordered Dithering (8x8 Bayer) 적용 중...")
        data = ordered_dither(img_array)
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
  2: 빨강 (Red)
  3: 노랑 (Yellow)
  4: 주황 (Orange)
  5: 초록 (Green)
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

