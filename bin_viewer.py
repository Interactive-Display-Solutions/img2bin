#!/usr/bin/env python3
"""
BIN 파일 검증 및 미리보기 도구

사용법:
    python3 bin_viewer.py input.bin [output.jpg]
"""

import sys
import numpy as np
from PIL import Image
from collections import Counter
import argparse

# EPD 6색 팔레트
EPD_PALETTE = {
    0: (0, 0, 0),        # 검정
    1: (255, 255, 255),  # 흰색
    2: (255, 0, 0),      # 빨강
    3: (255, 255, 0),    # 노랑
    4: (255, 128, 0),    # 주황
    5: (0, 255, 0),      # 초록
}

COLOR_NAMES = {
    0: "검정",
    1: "흰색",
    2: "빨강",
    3: "노랑",
    4: "주황",
    5: "초록"
}

TARGET_WIDTH = 1200
TARGET_HEIGHT = 1600
EXPECTED_SIZE = TARGET_WIDTH * TARGET_HEIGHT


def validate_and_preview(bin_file, output_jpg=None):
    """BIN 파일 검증 및 미리보기 생성"""
    
    print(f"BIN 파일 로딩: {bin_file}")
    print("=" * 50)
    
    # 파일 읽기
    with open(bin_file, 'rb') as f:
        data = f.read()
    
    # 파일 크기 검증
    file_size = len(data)
    print(f"\n파일 크기: {file_size:,} bytes")
    print(f"예상 크기: {EXPECTED_SIZE:,} bytes ({TARGET_WIDTH}x{TARGET_HEIGHT})")
    
    if file_size != EXPECTED_SIZE:
        print(f"⚠️  경고: 파일 크기가 예상과 다릅니다!")
        if file_size < EXPECTED_SIZE:
            print(f"   부족: {EXPECTED_SIZE - file_size:,} bytes")
        else:
            print(f"   초과: {file_size - EXPECTED_SIZE:,} bytes")
        return False
    else:
        print("✓ 파일 크기 정상")
    
    # numpy 배열로 변환
    pixel_data = np.frombuffer(data, dtype=np.uint8)
    
    # 픽셀 값 범위 확인
    min_val = pixel_data.min()
    max_val = pixel_data.max()
    print(f"\n픽셀 값 범위: {min_val} ~ {max_val}")
    
    if min_val < 0 or max_val > 5:
        print(f"⚠️  경고: 유효하지 않은 픽셀 값이 있습니다!")
        invalid_count = np.sum((pixel_data < 0) | (pixel_data > 5))
        print(f"   유효하지 않은 픽셀: {invalid_count:,}개")
    else:
        print("✓ 픽셀 값 범위 정상 (0-5)")
    
    # 색상 분포 분석
    counter = Counter(pixel_data)
    print(f"\n색상 분포:")
    print("-" * 50)
    print(f"{'인덱스':<8} {'색상':<10} {'픽셀 수':<15} {'비율':<10}")
    print("-" * 50)
    
    for i in range(6):
        count = counter.get(i, 0)
        percentage = (count / len(pixel_data)) * 100
        color_name = COLOR_NAMES.get(i, "알 수 없음")
        print(f"{i:<8} {color_name:<10} {count:<15,} {percentage:>6.2f}%")
    
    # 유효하지 않은 값 확인
    invalid_values = set(counter.keys()) - set(range(6))
    if invalid_values:
        print("\n⚠️  유효하지 않은 색상 인덱스:")
        for val in sorted(invalid_values):
            count = counter[val]
            print(f"   {val}: {count:,}개")
    
    # 미리보기 생성
    if output_jpg or output_jpg is None:
        img_data = pixel_data.reshape((TARGET_HEIGHT, TARGET_WIDTH))
        rgb_img = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        
        for color_idx, rgb in EPD_PALETTE.items():
            mask = img_data == color_idx
            rgb_img[mask] = rgb
        
        # 출력 파일명 결정
        if output_jpg is None:
            output_jpg = bin_file.rsplit('.', 1)[0] + '_preview.jpg'
        
        img = Image.fromarray(rgb_img)
        img.save(output_jpg, quality=85)
        print(f"\n✓ 미리보기 저장: {output_jpg}")
    
    print("\n" + "=" * 50)
    print("검증 완료!")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='EPD BIN 파일 검증 및 미리보기 생성',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # BIN 파일 검증하고 미리보기 생성
  python3 bin_viewer.py output.bin
  
  # 특정 이름으로 미리보기 저장
  python3 bin_viewer.py output.bin preview.jpg
  
  # 미리보기 없이 검증만 수행
  python3 bin_viewer.py output.bin --no-preview
        """
    )
    
    parser.add_argument('bin_file', help='검증할 BIN 파일 경로')
    parser.add_argument('output_jpg', nargs='?', help='미리보기 JPG 파일 경로 (선택)')
    parser.add_argument('--no-preview', action='store_true',
                        help='미리보기 생성하지 않음')
    
    args = parser.parse_args()
    
    output_jpg = None if args.no_preview else args.output_jpg
    
    try:
        success = validate_and_preview(args.bin_file, output_jpg)
        sys.exit(0 if success else 1)
    except FileNotFoundError:
        print(f"\n오류: 파일을 찾을 수 없습니다 - {args.bin_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


