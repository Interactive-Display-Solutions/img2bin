# 빠른 시작 가이드

ESP32-S3 ok-dev03a EPD용 JPG to BIN 변환 도구

## 5분 안에 시작하기

### 1단계: 의존성 설치

```bash
pip3 install -r requirements.txt
```

### 2단계: 이미지 변환

```bash
# 방법 1: 디더링 사용 (부드러운 색상)
python3 jpg_to_bin.py your_image.jpg output.bin

# 방법 2: 디더링 없음 (선명한 색상)
python3 jpg_to_bin.py your_image.jpg output.bin --no-dither
```

### 3단계: 결과 확인

```bash
python3 bin_viewer.py output.bin
```

## 자동 예제 실행

```bash
bash example.sh
```

## 주요 기능

✅ **1200x1600 해상도** - EPD 최적화  
✅ **6색 팔레트** - 검정, 흰색, 빨강, 노랑, 주황, 초록  
✅ **디더링 옵션** - Floyd-Steinberg 알고리즘  
✅ **검증 도구** - 파일 무결성 확인  
✅ **미리보기** - 결과 시각화

## 출력 형식

- **파일 크기**: 1,920,000 bytes
- **형식**: Raw binary (헤더 없음)
- **인코딩**: 픽셀당 1 byte (색상 인덱스 0-5)

## 도움말

```bash
python3 jpg_to_bin.py --help
python3 bin_viewer.py --help
```

더 자세한 정보는 `README.md`를 참고하세요.






