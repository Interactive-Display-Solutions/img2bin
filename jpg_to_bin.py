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

# EPD 6색 팔레트 정의 (ACeP/Spectra 6)
# 각 색상의 RGB 값 (ok-dev03a 보드용)
EPD_PALETTE = {
    0: (0, 0, 0),        # 검정
    1: (255, 255, 255),  # 흰색
    2: (255, 0, 0),      # 빨강
    3: (255, 255, 0),    # 노랑
    4: (255, 128, 0),    # 주황
    5: (0, 255, 0),      # 초록
}

# 대상 이미지 크기
TARGET_WIDTH = 1200
TARGET_HEIGHT = 1600


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
    
    return nearest_index


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
    
    # PIL을 사용한 팔레트 양자화 (더 빠른 방법)
    # EPD 팔레트로 이미지 생성
    palette_img = Image.new('P', (1, 1))
    palette_data = []
    for i in range(6):
        palette_data.extend(EPD_PALETTE[i])
    # 나머지 팔레트는 0으로 채움 (256색까지)
    palette_data.extend([0] * (256 - 6) * 3)
    palette_img.putpalette(palette_data)
    
    # 이미지를 팔레트 모드로 변환
    dither_mode = Image.Dither.FLOYDSTEINBERG if use_dithering else Image.Dither.NONE
    quantized = canvas.quantize(colors=6, palette=palette_img, dither=dither_mode)
    
    print("팔레트 양자화 완료")
    
    # numpy 배열로 변환
    data = np.array(quantized, dtype=np.uint8)
    
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

