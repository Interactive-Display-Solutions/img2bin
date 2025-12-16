#!/usr/bin/env python3
"""
epdoptimize 스타일의 디더링 알고리즘 구현
참고: https://github.com/Utzel-Butzel/epdoptimize
"""

import numpy as np
from PIL import Image
import os

# EPD 팔레트 (Spectra 6)
EPD_PALETTE = {
    0: (0x00, 0x00, 0x00),    # 검정 (black)
    1: (0xFF, 0xFF, 0xFF),    # 흰색 (white)
    2: (0xFF, 0xFF, 0x00),    # 노랑 (yellow)
    3: (0xFF, 0x00, 0x00),    # 빨강 (red)
    4: (0x00, 0x00, 0xFF),    # 파랑 (blue)
    5: (0x00, 0xFF, 0x00),    # 초록 (green)
}

PALETTE_ARRAY = np.array([EPD_PALETTE[i] for i in range(6)], dtype=np.float32)

def rgb_to_nearest_palette(rgb):
    """RGB를 가장 가까운 팔레트 색상으로 매핑"""
    distances = np.sum((PALETTE_ARRAY - rgb) ** 2, axis=1)
    return np.argmin(distances)

def find_nearest_palette_color(rgb):
    """가장 가까운 팔레트 색상과 인덱스 반환"""
    distances = np.sum((PALETTE_ARRAY - rgb) ** 2, axis=1)
    idx = np.argmin(distances)
    return idx, PALETTE_ARRAY[idx]

# Error Diffusion Matrices (epdoptimize 참고)
ERROR_DIFFUSION_MATRICES = {
    'floydSteinberg': {
        'matrix': np.array([
            [0, 0, 7],
            [3, 5, 1]
        ]) / 16.0,
        'offset': (0, 1)
    },
    'falseFloydSteinberg': {
        'matrix': np.array([
            [0, 0, 3],
            [0, 3, 2]
        ]) / 8.0,
        'offset': (0, 1)
    },
    'jarvis': {
        'matrix': np.array([
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1]
        ]) / 48.0,
        'offset': (0, 2)
    },
    'stucki': {
        'matrix': np.array([
            [0, 0, 0, 8, 4],
            [2, 4, 8, 4, 2],
            [1, 2, 4, 2, 1]
        ]) / 42.0,
        'offset': (0, 2)
    },
    'burkes': {
        'matrix': np.array([
            [0, 0, 0, 8, 4],
            [2, 4, 8, 4, 2]
        ]) / 32.0,
        'offset': (0, 1)
    },
    'sierra3': {
        'matrix': np.array([
            [0, 0, 0, 5, 3],
            [2, 4, 5, 4, 2],
            [0, 2, 3, 2, 0]
        ]) / 32.0,
        'offset': (0, 2)
    },
    'sierra2': {
        'matrix': np.array([
            [0, 0, 0, 4, 3],
            [1, 2, 3, 2, 1]
        ]) / 16.0,
        'offset': (0, 1)
    },
    'sierra2-4a': {
        'matrix': np.array([
            [0, 0, 2],
            [1, 1, 0]
        ]) / 4.0,
        'offset': (0, 1)
    }
}

def error_diffusion_dither(img_array, algorithm='floydSteinberg', serpentine=False):
    """
    Error Diffusion 디더링 (epdoptimize 스타일)
    
    Args:
        img_array: HxWx3 uint8 RGB 이미지
        algorithm: 'floydSteinberg', 'jarvis', 'stucki', 'burkes', 'sierra3', 'sierra2', 'sierra2-4a'
        serpentine: True면 행마다 스캔 방향 반대
    
    Returns:
        HxW uint8 배열 (팔레트 인덱스 0-5)
    """
    if algorithm not in ERROR_DIFFUSION_MATRICES:
        algorithm = 'floydSteinberg'
    
    config = ERROR_DIFFUSION_MATRICES[algorithm]
    matrix = config['matrix']
    offset_y, offset_x = config['offset']
    
    H, W = img_array.shape[:2]
    result = np.zeros((H, W), dtype=np.uint8)
    
    # float 배열로 작업 (오차 계산용)
    working = img_array.astype(np.float32)
    
    for y in range(H):
        # Serpentine: 짝수 행은 왼쪽->오른쪽, 홀수 행은 오른쪽->왼쪽
        x_range = range(W) if not serpentine or y % 2 == 0 else range(W - 1, -1, -1)
        
        for x in x_range:
            # 현재 픽셀 RGB
            rgb = working[y, x].copy()
            
            # 가장 가까운 팔레트 색상 찾기
            nearest_idx, nearest_rgb = find_nearest_palette_color(rgb)
            
            # 결과 저장
            result[y, x] = nearest_idx
            
            # 오차 계산
            error = rgb - nearest_rgb
            
            # 오차를 주변 픽셀에 분산
            matrix_h, matrix_w = matrix.shape
            start_y = y
            start_x = x + offset_x if not serpentine or y % 2 == 0 else x - offset_x
            
            for dy in range(matrix_h):
                for dx in range(matrix_w):
                    weight = matrix[dy, dx]
                    if weight == 0:
                        continue
                    
                    # Serpentine 고려
                    if serpentine and y % 2 == 1:
                        target_x = start_x - dx
                    else:
                        target_x = start_x + dx - offset_x
                    
                    target_y = start_y + dy
                    
                    # 범위 체크
                    if 0 <= target_y < H and 0 <= target_x < W:
                        working[target_y, target_x] += error * weight
    
    return result

def bayer_ordered_dither(img_array, matrix_size=4):
    """
    Bayer Ordered Dithering (epdoptimize 스타일)
    
    Args:
        img_array: HxWx3 uint8 RGB 이미지
        matrix_size: Bayer 매트릭스 크기 (4, 8 등)
    
    Returns:
        HxW uint8 배열 (팔레트 인덱스 0-5)
    """
    H, W = img_array.shape[:2]
    result = np.zeros((H, W), dtype=np.uint8)
    
    # Bayer 매트릭스 생성
    def generate_bayer_matrix(n):
        """n x n Bayer 매트릭스 생성"""
        if n == 1:
            return np.array([[0]])
        elif n == 2:
            return np.array([[0, 2], [3, 1]])
        else:
            # 재귀적으로 생성
            smaller = generate_bayer_matrix(n // 2)
            matrix = np.zeros((n, n), dtype=np.float32)
            for i in range(2):
                for j in range(2):
                    matrix[i*n//2:(i+1)*n//2, j*n//2:(j+1)*n//2] = (
                        smaller * 4 + np.array([[0, 2], [3, 1]])[i, j]
                    )
            return matrix / (n * n)
    
    bayer_matrix = generate_bayer_matrix(matrix_size)
    threshold = bayer_matrix * 255.0
    
    for y in range(H):
        for x in range(W):
            rgb = img_array[y, x].astype(np.float32)
            
            # Bayer 매트릭스 값
            threshold_val = threshold[y % matrix_size, x % matrix_size]
            
            # 각 채널에 threshold 적용
            dithered_rgb = np.where(rgb >= threshold_val, 255, 0).astype(np.uint8)
            
            # 가장 가까운 팔레트 색상 찾기
            nearest_idx = rgb_to_nearest_palette(dithered_rgb)
            result[y, x] = nearest_idx
    
    return result

def quantization_only(img_array):
    """
    양자화만 수행 (디더링 없음)
    
    Args:
        img_array: HxWx3 uint8 RGB 이미지
    
    Returns:
        HxW uint8 배열 (팔레트 인덱스 0-5)
    """
    H, W = img_array.shape[:2]
    result = np.zeros((H, W), dtype=np.uint8)
    
    for y in range(H):
        for x in range(W):
            rgb = img_array[y, x]
            nearest_idx = rgb_to_nearest_palette(rgb)
            result[y, x] = nearest_idx
    
    return result

def compare_with_test_bin(result, bin_file_path):
    """결과를 test bin 파일과 비교"""
    with open(bin_file_path, 'rb') as f:
        expected = np.frombuffer(f.read(), dtype=np.uint8)
    
    if len(result.flatten()) != len(expected):
        return 100.0, 0, len(expected)
    
    diff = np.sum(result.flatten() != expected)
    error_rate = (diff / len(expected)) * 100.0
    return error_rate, diff, len(expected)

def test_all_algorithms():
    """모든 알고리즘을 test 폴더의 파일들과 비교"""
    import glob
    import os
    
    test_dir = "test"
    jpg_files = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
    
    algorithms = {
        'floydSteinberg': lambda img: error_diffusion_dither(img, 'floydSteinberg'),
        'floydSteinberg_serpentine': lambda img: error_diffusion_dither(img, 'floydSteinberg', serpentine=True),
        'jarvis': lambda img: error_diffusion_dither(img, 'jarvis'),
        'stucki': lambda img: error_diffusion_dither(img, 'stucki'),
        'burkes': lambda img: error_diffusion_dither(img, 'burkes'),
        'sierra3': lambda img: error_diffusion_dither(img, 'sierra3'),
        'sierra2': lambda img: error_diffusion_dither(img, 'sierra2'),
        'sierra2-4a': lambda img: error_diffusion_dither(img, 'sierra2-4a'),
        'bayer4': lambda img: bayer_ordered_dither(img, 4),
        'bayer8': lambda img: bayer_ordered_dither(img, 8),
        'quantization_only': quantization_only,
    }
    
    results_summary = {}
    
    print("="*80)
    print("Testing epdoptimize-style algorithms against test bin files")
    print("="*80)
    
    for alg_name, alg_func in algorithms.items():
        print(f"\n{'='*80}")
        print(f"Algorithm: {alg_name}")
        print(f"{'='*80}")
        
        total_error = 0.0
        total_pixels = 0
        file_results = []
        
        for jpg_file in jpg_files:
            base_name = os.path.splitext(os.path.basename(jpg_file))[0]
            bin_file = os.path.join(test_dir, f"{base_name}bin")
            
            if not os.path.exists(bin_file):
                continue
            
            try:
                # 이미지 로드 및 리사이즈
                img = Image.open(jpg_file).convert('RGB')
                img.thumbnail((1200, 1600), Image.Resampling.LANCZOS)
                canvas = Image.new('RGB', (1200, 1600), (255, 255, 255))
                offset_x = (1200 - img.width) // 2
                offset_y = (1600 - img.height) // 2
                canvas.paste(img, (offset_x, offset_y))
                img_array = np.array(canvas, dtype=np.uint8)
                
                # 디더링 적용
                result = alg_func(img_array)
                
                # 비교
                error_rate, diff, total = compare_with_test_bin(result, bin_file)
                
                total_error += diff
                total_pixels += total
                file_results.append((base_name, error_rate, diff, total))
                
                status = "✅" if error_rate < 1.0 else "❌"
                print(f"  {status} {base_name:20s}: {error_rate:6.2f}% ({diff:8d}/{total:8d})")
                
            except Exception as e:
                print(f"  ❌ {base_name}: Error - {str(e)}")
                file_results.append((base_name, 100.0, 0, 0))
        
        overall_error = (total_error / total_pixels * 100.0) if total_pixels > 0 else 100.0
        results_summary[alg_name] = {
            'overall_error': overall_error,
            'file_results': file_results
        }
        
        print(f"\n  Overall error rate: {overall_error:.2f}%")
    
    # 최종 요약
    print(f"\n{'='*80}")
    print("SUMMARY - Best algorithms (sorted by error rate)")
    print(f"{'='*80}")
    
    sorted_results = sorted(results_summary.items(), key=lambda x: x[1]['overall_error'])
    
    for i, (alg_name, data) in enumerate(sorted_results[:5], 1):
        print(f"{i}. {alg_name:25s}: {data['overall_error']:6.2f}%")
    
    print(f"{'='*80}\n")
    
    return results_summary

if __name__ == '__main__':
    test_all_algorithms()

