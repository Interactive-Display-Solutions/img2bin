# JPG to BIN Converter for E Ink Spectra6

ESP32-S3 MCU를 사용하는 ok-dev03a 보드의 EPD(전자 종이 디스플레이)용 이미지 변환 도구입니다.

## 개요

이 도구는 JPG/PNG 이미지를 1200x1600 해상도의 6색 EPD 디스플레이용 바이너리 파일로 변환합니다. 웹 인터페이스를 통해 브라우저에서 직접 사용할 수 있으며, 다양한 디더링 알고리즘을 지원합니다.

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

## 주요 기능

### 🎨 다양한 디더링 알고리즘
- **E Ink Default**: E Ink 공식 알고리즘 (8x8 Bayer 매트릭스)
- **Spectra 6 Vivid**: OKLab 색 공간 기반 최적화 알고리즘
- **Floyd-Steinberg**: 오차 확산 디더링 (기본 및 Serpentine)
- **Jarvis, Stucki, Burkes**: 다양한 오차 확산 방법
- **Sierra 시리즈**: Sierra-3, Sierra-2, Sierra-2-4A
- **Bayer Ordered**: 4x4, 8x8 Bayer 매트릭스
- **Quantization Only**: 디더링 없이 양자화만

### 🖼️ 실시간 이미지 조정
- **Contrast (대비)**: 0.5 ~ 2.0
- **Brightness (밝기)**: 0.5 ~ 2.0
- **Saturation (채도)**: 0.0 ~ 2.0
- **Hue Shift (색상 회전)**: -180° ~ +180°
- **Smoothing (부드러움)**: 0.0 ~ 2.0 (점 패턴 감소)

### 📊 참조 파일 비교
- 같은 폴더의 참조 BIN 파일과 자동 비교
- 오류율 계산 및 최적 알고리즘 추천

## 설치

필요한 Python 패키지를 설치합니다:

```bash
pip3 install -r requirements.txt
```

또는 직접 설치:

```bash
pip3 install Pillow numpy Flask
```

## 사용법

### 웹 인터페이스 (추천)

웹 브라우저에서 사용할 수 있는 GUI 인터페이스가 제공됩니다:

```bash
python3 web_viewer.py
```

그 다음 브라우저에서 `http://localhost:8000` 접속

#### 웹 인터페이스 기능

1. **JPG → BIN 변환**
   - 드래그 앤 드롭으로 이미지 업로드
   - 다양한 디더링 알고리즘 선택
   - 실시간 미리보기 및 통계 확인
   - BIN 파일 다운로드

2. **실시간 조정 및 미리보기**
   - 이미지 조정 슬라이더 (Contrast, Brightness, Saturation, Hue, Smoothing)
   - "Update Preview" 버튼으로 실시간 확인
   - 조정된 설정으로 BIN 파일 다운로드

3. **알고리즘 테스트**
   - 선택한 알고리즘으로 변환
   - 참조 BIN 파일과 자동 비교
   - 오류율 표시

4. **BIN 파일 미리보기**
   - BIN 파일 업로드하여 미리보기 확인
   - 색상 분포 통계 확인

### 명령줄 사용법

#### 기본 사용법 (디더링 적용)

```bash
python3 jpg_to_bin.py input.jpg output.bin
```

#### 디더링 없이 변환

```bash
python3 jpg_to_bin.py input.jpg output.bin --no-dither
```

#### BIN 파일 검증 및 미리보기

```bash
# 검증하고 미리보기 생성
python3 bin_viewer.py output.bin

# 특정 이름으로 미리보기 저장
python3 bin_viewer.py output.bin preview.jpg
```

## 출력 형식

변환된 BIN 파일은 다음 형식을 따릅니다:

- **파일 크기**: 1,920,000 바이트 (1200 × 1600 픽셀)
- **형식**: 헤더 없는 순수 픽셀 데이터
- **픽셀 인코딩**: 각 픽셀당 1바이트 (색상 인덱스 0-5)
- **픽셀 순서**: 좌→우, 위→아래 (row-major order)

## 이미지 처리 과정

1. **입력 이미지 로드**: JPG/PNG 파일을 RGB 모드로 로드
2. **리사이즈**: 1200x1600 해상도로 리사이즈 (비율 유지)
   - 원본 비율을 유지하며 캔버스에 맞춤
   - 남은 공간은 흰색으로 채움
3. **이미지 조정** (선택사항): Contrast, Brightness, Saturation, Hue, Smoothing 적용
4. **색상 양자화**: 6색 팔레트로 변환
   - 선택한 디더링 알고리즘 적용
5. **바이너리 출력**: 픽셀 데이터를 바이너리 파일로 저장

## 디더링 알고리즘 설명

### E Ink Default (기본값)
- E Ink 공식 알고리즘
- 8x8 Bayer 매트릭스 사용
- Magenta/Yellow 색상 보정 포함

### Spectra 6 Vivid
- OKLab 색 공간 기반
- Chroma thresholding 및 exaggeration
- Hue snapping
- Blue-noise ordered dithering

### Error Diffusion 알고리즘들
- **Floyd-Steinberg**: 가장 널리 사용되는 오차 확산
- **Serpentine**: 지그재그 스캔으로 패턴 감소
- **Jarvis, Stucki, Burkes**: 다양한 커널 크기
- **Sierra 시리즈**: 더 부드러운 결과

### Ordered Dithering
- **Bayer 4x4/8x8**: 규칙적인 패턴
- 빠른 처리 속도

## 웹 배포

이 프로젝트는 Render.com, Railway.app 등에서 바로 배포할 수 있습니다.

### Render.com 배포 (추천)

1. [Render.com](https://render.com)에 가입 (GitHub 계정으로 로그인)
2. "New +" → "Web Service" 클릭
3. GitHub 저장소 연결
4. 자동으로 설정이 감지됩니다 (`render.yaml` 파일 사용)
5. 배포 완료!

자세한 배포 방법은 [DEPLOY.md](DEPLOY.md)를 참조하세요.

## 파일 구조

```
img2bin/
├── jpg_to_bin.py          # 메인 변환 스크립트
├── bin_viewer.py          # BIN 파일 검증 및 미리보기 도구
├── web_viewer.py          # 웹 인터페이스 (Flask)
├── epdoptimize_dither.py  # 다양한 디더링 알고리즘
├── image_adjustments.py   # 이미지 조정 함수
├── templates/
│   └── index.html         # 웹 인터페이스 HTML
├── requirements.txt        # Python 의존성
├── render.yaml            # Render.com 배포 설정
├── Procfile               # Heroku/Railway 배포 설정
├── runtime.txt            # Python 버전 지정
├── DEPLOY.md              # 배포 가이드
└── README.md              # 이 파일
```

## 기술 명세

- **대상 보드**: ok-dev03a (ESP32-S3 MCU)
- **디스플레이**: 1200x1600 ACeP 6색 EPD
- **Python 버전**: 3.7 이상
- **의존성**: 
  - Pillow (PIL) >= 10.0.0
  - NumPy >= 1.24.0
  - Flask >= 3.0.0

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

1. 다른 디더링 알고리즘을 시도해보세요
2. 이미지 조정 옵션을 사용해보세요 (Contrast, Brightness 등)
3. Smoothing 옵션으로 점 패턴을 줄여보세요

## 라이선스

이 프로젝트는 자유롭게 사용, 수정, 배포할 수 있습니다.

## 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!
