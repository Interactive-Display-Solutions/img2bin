# JPG to BIN Converter for ESP32-S3 EPD

ESP32-S3 MCU를 사용하는 ok-dev03a 보드의 EPD(전자 종이 디스플레이)용 이미지 변환 도구입니다.

## 개요

이 도구는 JPG 이미지를 1200x1600 해상도의 6색 EPD 디스플레이용 바이너리 파일로 변환합니다.

### 지원 색상 팔레트

ok-dev03a 보드의 ACeP(Advanced Color ePaper) 디스플레이는 다음 6가지 색상을 지원합니다:

| 인덱스 | 색상 | RGB 값 |
|--------|------|---------|
| 0 | 검정 (Black) | (0, 0, 0) |
| 1 | 흰색 (White) | (255, 255, 255) |
| 2 | 노랑 (Yellow) | (255, 255, 0) |
| 3 | 빨강 (Red) | (255, 0, 0) |
| 4 | 파랑 (Blue) | (0, 0, 255) |
| 5 | 초록 (Green) | (0, 255, 0) |

## 설치

필요한 Python 패키지를 설치합니다:

```bash
pip3 install -r requirements.txt
```

또는 직접 설치:

```bash
pip3 install Pillow numpy
```

## 사용법

### 기본 사용법 (디더링 적용)

Floyd-Steinberg 디더링을 사용하여 부드러운 색상 전환을 생성합니다:

```bash
python3 jpg_to_bin.py input.jpg output.bin
```

### 디더링 없이 변환

더 선명한 색상을 원하는 경우 디더링을 비활성화할 수 있습니다:

```bash
python3 jpg_to_bin.py input.jpg output.bin --no-dither
```

### BIN 파일 검증 및 미리보기

변환된 BIN 파일을 검증하고 미리보기를 생성할 수 있습니다:

```bash
# 검증하고 미리보기 생성
python3 bin_viewer.py output.bin

# 특정 이름으로 미리보기 저장
python3 bin_viewer.py output.bin preview.jpg

# 검증만 수행 (미리보기 없이)
python3 bin_viewer.py output.bin --no-preview
```

### 예제

```bash
# 샘플 이미지 변환
python3 jpg_to_bin.py demo5.jpg demo5.bin

# 나만의 이미지 변환
python3 jpg_to_bin.py my_photo.jpg my_photo.bin --no-dither

# 변환 결과 확인
python3 bin_viewer.py my_photo.bin

# 자동 예제 실행
bash example.sh
```

### 웹 인터페이스 사용

웹 브라우저에서 사용할 수 있는 GUI 인터페이스가 제공됩니다:

```bash
python3 web_viewer.py
```

그 다음 브라우저에서 `http://localhost:8000` 접속

웹 인터페이스에서는:
- JPG/PNG 파일을 드래그 앤 드롭하여 BIN으로 변환
- BIN 파일을 업로드하여 미리보기 확인
- 디더링 옵션 선택 가능
- 색상 분포 통계 확인

### 도움말

```bash
python3 jpg_to_bin.py --help
python3 bin_viewer.py --help
```

## 출력 형식

변환된 BIN 파일은 다음 형식을 따릅니다:

- **파일 크기**: 1,920,000 바이트 (1200 × 1600 픽셀)
- **형식**: 헤더 없는 순수 픽셀 데이터
- **픽셀 인코딩**: 각 픽셀당 1바이트 (색상 인덱스 0-5)
- **픽셀 순서**: 좌→우, 위→아래 (row-major order)

## 이미지 처리 과정

1. **입력 이미지 로드**: JPG 파일을 RGB 모드로 로드
2. **리사이즈**: 1200x1600 해상도로 리사이즈 (비율 유지)
   - 원본 비율을 유지하며 캔버스에 맞춤
   - 남은 공간은 흰색으로 채움
3. **색상 양자화**: 6색 팔레트로 변환
   - Floyd-Steinberg 디더링 (기본값)
   - 또는 디더링 없이 변환 (`--no-dither`)
4. **바이너리 출력**: 픽셀 데이터를 바이너리 파일로 저장

## ESP32-S3 보드에서 사용하기

생성된 BIN 파일을 ESP32-S3 보드로 전송하여 EPD에 표시할 수 있습니다:

```c
// ESP32-S3 펌웨어 예제 (의사 코드)
#include <stdio.h>
#include <stdint.h>

#define EPD_WIDTH  1200
#define EPD_HEIGHT 1600

void display_image(const uint8_t* image_data) {
    // EPD 초기화
    epd_init();
    
    // 이미지 데이터를 EPD로 전송
    for (int y = 0; y < EPD_HEIGHT; y++) {
        for (int x = 0; x < EPD_WIDTH; x++) {
            uint8_t color_index = image_data[y * EPD_WIDTH + x];
            epd_set_pixel(x, y, color_index);
        }
    }
    
    // EPD 화면 업데이트
    epd_refresh();
}
```

## 팁과 권장사항

### 이미지 준비

1. **해상도**: 원본 이미지가 1200x1600에 가까울수록 좋은 결과를 얻을 수 있습니다
2. **색상**: 6색 팔레트에 맞는 색상을 사용한 이미지가 가장 잘 표현됩니다
3. **명암**: 높은 명암비의 이미지가 EPD에서 더 잘 보입니다

### 디더링 선택

- **디더링 사용 (기본값)**: 그라데이션이나 부드러운 색상 전환이 있는 이미지에 권장
- **디더링 없음**: 텍스트, 로고, 선명한 경계가 있는 그래픽에 권장

### 최적의 결과를 위한 전처리

더 나은 결과를 위해 GIMP나 Photoshop 같은 이미지 편집 도구에서:
1. 이미지를 1200x1600으로 리사이즈
2. 대비와 채도를 조정
3. 6색 팔레트를 염두에 두고 색상 조정

## 파일 구조

```
jpg_2_bin/
├── jpg_to_bin.py          # 메인 변환 스크립트
├── bin_viewer.py          # BIN 파일 검증 및 미리보기 도구
├── web_viewer.py          # 웹 인터페이스 (Flask)
├── templates/
│   └── index.html         # 웹 인터페이스 HTML
├── example.sh             # 자동 예제 실행 스크립트
├── requirements.txt        # Python 의존성
├── render.yaml            # Render.com 배포 설정
├── Procfile               # Heroku/Railway 배포 설정
├── runtime.txt            # Python 버전 지정
├── DEPLOY.md              # 배포 가이드
├── README.md              # 이 파일
├── .gitignore             # Git 무시 파일 목록
├── demo5.jpg              # 샘플 입력 이미지
└── demo5bin               # 샘플 출력 파일
```

## 기술 명세

- **대상 보드**: ok-dev03a (ESP32-S3 MCU)
- **디스플레이**: 1200x1600 ACeP 6색 EPD
- **JPG 인코딩**: Baseline encoding
- **Python 버전**: 3.7 이상
- **의존성**: 
  - Pillow (PIL) >= 10.0.0
  - NumPy >= 1.24.0

## 문제 해결

### "ModuleNotFoundError: No module named 'PIL'"

```bash
pip3 install Pillow
```

### "numpy module not found"

```bash
pip3 install numpy
```

### 변환된 이미지가 예상과 다름

1. `--no-dither` 옵션을 시도해보세요
2. 원본 이미지의 색상과 대비를 조정해보세요
3. 6색 팔레트에 맞게 원본 이미지를 사전 처리하세요

## 웹 배포

이 프로젝트를 GitHub에 푸시하면 여러 플랫폼에서 바로 배포할 수 있습니다.

### 빠른 배포 (Render.com 추천)

1. [Render.com](https://render.com)에 가입 (GitHub 계정으로 로그인)
2. "New +" → "Web Service" 클릭
3. GitHub 저장소 연결
4. 자동으로 설정이 감지됩니다 (또는 `render.yaml` 파일 사용)
5. 배포 완료!

자세한 배포 방법은 [DEPLOY.md](DEPLOY.md)를 참조하세요.

## 라이선스

이 프로젝트는 자유롭게 사용, 수정, 배포할 수 있습니다.

## 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

