# 불꽃 이미지 변환 방법 🔥

## 방법 1: 드래그 앤 드롭

1. 불꽃 이미지 파일을 이 폴더로 드래그 앤 드롭
2. 터미널에서 실행:

```bash
cd "/Users/sangwonchae/Library/CloudStorage/OneDrive-InteractiveDisplaySolutions,Inc/Programming/jpg_2_bin"

# 이미지 파일명을 실제 파일명으로 변경하세요
python3 jpg_to_bin.py 불꽃이미지.jpg output.bin
```

## 방법 2: 명령어로 직접 변환

```bash
# 디더링 사용 (부드러운 색상 전환)
python3 jpg_to_bin.py [이미지파일].jpg output_dither.bin

# 디더링 없음 (선명한 색상)
python3 jpg_to_bin.py [이미지파일].jpg output_no_dither.bin --no-dither
```

## 방법 3: 자동 스크립트 사용

이미지를 `real_fire.jpg`로 저장한 후:

```bash
bash convert_fire.sh
```

## 예제

현재 폴더에 `my_fire.jpg`가 있다면:

```bash
# 변환
python3 jpg_to_bin.py my_fire.jpg my_fire.bin

# 결과 확인
python3 bin_viewer.py my_fire.bin
```

## 빠른 테스트

샘플 이미지로 테스트:

```bash
python3 jpg_to_bin.py demo5.jpg test_output.bin
python3 bin_viewer.py test_output.bin
```





