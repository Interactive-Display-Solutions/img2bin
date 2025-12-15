#!/bin/bash

echo "=========================================="
echo "🔥 불꽃 이미지 변환"
echo "=========================================="
echo ""

# 이미지 파일명 (사용자가 업로드한 이미지)
IMAGE_FILE="real_fire.jpg"

if [ ! -f "$IMAGE_FILE" ]; then
    echo "⚠️  $IMAGE_FILE 파일을 찾을 수 없습니다."
    echo ""
    echo "다음 단계를 따라주세요:"
    echo "1. 불꽃 이미지를 이 폴더에 저장하세요"
    echo "2. 파일명을 'real_fire.jpg'로 지정하세요"
    echo "3. 다시 이 스크립트를 실행하세요"
    echo ""
    echo "또는 다음 명령어를 직접 실행하세요:"
    echo "  python3 jpg_to_bin.py [이미지파일] output.bin"
    exit 1
fi

echo "📁 입력 파일: $IMAGE_FILE"
echo ""

# 1. 디더링 사용 변환
echo "1️⃣  디더링 사용 변환..."
python3 jpg_to_bin.py "$IMAGE_FILE" real_fire_dither.bin
echo ""

# 2. 디더링 없이 변환
echo "2️⃣  디더링 없이 변환..."
python3 jpg_to_bin.py "$IMAGE_FILE" real_fire_no_dither.bin --no-dither
echo ""

# 3. 결과 검증
echo "3️⃣  결과 검증..."
echo ""
python3 bin_viewer.py real_fire_dither.bin
echo ""

echo "=========================================="
echo "✅ 변환 완료!"
echo ""
echo "생성된 파일:"
echo "  📦 real_fire_dither.bin       (디더링 사용)"
echo "  📦 real_fire_no_dither.bin    (디더링 없음)"
echo "  🖼️  미리보기 이미지들"
echo ""
echo "ESP32-S3 보드로 전송하여 EPD에 표시할 수 있습니다!"
echo "=========================================="





