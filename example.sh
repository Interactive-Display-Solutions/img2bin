#!/bin/bash

# JPG to BIN Converter - 사용 예제 스크립트

echo "=========================================="
echo "JPG to BIN Converter 예제"
echo "ESP32-S3 ok-dev03a EPD용"
echo "=========================================="
echo ""

# 의존성 확인
echo "1. Python 패키지 설치 확인..."
if ! python3 -c "import PIL" 2>/dev/null; then
    echo "   ⚠️  Pillow가 설치되지 않았습니다. 설치 중..."
    pip3 install -r requirements.txt
else
    echo "   ✓ 필요한 패키지가 모두 설치되어 있습니다."
fi

echo ""
echo "2. 샘플 이미지 변환..."
echo ""

# 디더링 사용
echo "   [1/2] 디더링 사용 변환..."
python3 jpg_to_bin.py demo5.jpg output_with_dither.bin
echo ""

# 디더링 없이
echo "   [2/2] 디더링 없이 변환..."
python3 jpg_to_bin.py demo5.jpg output_no_dither.bin --no-dither
echo ""

echo "=========================================="
echo "✓ 변환 완료!"
echo ""
echo "생성된 파일:"
echo "  - output_with_dither.bin   (디더링 사용)"
echo "  - output_no_dither.bin     (디더링 없음)"
echo ""
echo "파일 크기: $(ls -lh output_with_dither.bin | awk '{print $5}')"
echo ""
echo "이 파일들을 ESP32-S3 보드로 전송하여"
echo "EPD에 표시할 수 있습니다."
echo "=========================================="





