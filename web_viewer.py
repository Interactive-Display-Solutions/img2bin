#!/usr/bin/env python3
"""
JPG to BIN 변환기 웹 인터페이스

사용법:
    python3 web_viewer.py
    
그 다음 브라우저에서 http://localhost:8000 접속
"""

from flask import Flask, render_template, request, send_file, jsonify
import os
import io
import base64
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import tempfile

# jpg_to_bin과 bin_viewer 모듈 임포트
import jpg_to_bin
import bin_viewer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bin'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    """Convert JPG to BIN"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file format'}), 400
    
    try:
        use_dither = request.form.get('dither', 'true').lower() == 'true'
        
        # 임시 파일로 저장
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file.save(temp_input.name)
        temp_input.close()
        
        # BIN 파일 생성
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        temp_output.close()
        
        # 변환 실행
        jpg_to_bin.convert_jpg_to_bin(temp_input.name, temp_output.name, use_dithering=use_dither)
        
        # 미리보기 생성
        temp_preview = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_preview.close()
        
        bin_viewer.validate_and_preview(temp_output.name, temp_preview.name)
        
        # 미리보기 이미지를 base64로 인코딩
        with open(temp_preview.name, 'rb') as f:
            preview_data = base64.b64encode(f.read()).decode('utf-8')
        
        # BIN 파일 데이터
        with open(temp_output.name, 'rb') as f:
            bin_data = f.read()
        
        # 통계 정보
        pixel_data = np.frombuffer(bin_data, dtype=np.uint8)
        from collections import Counter
        counter = Counter(pixel_data)
        
        stats = {
            'file_size': len(bin_data),
            'colors': {}
        }
        
        for i in range(6):
            count = counter.get(i, 0)
            percentage = (count / len(pixel_data)) * 100
            stats['colors'][bin_viewer.COLOR_NAMES[i]] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        # 정리
        os.unlink(temp_input.name)
        os.unlink(temp_preview.name)
        
        # 원본 파일명에서 확장자 제거하고 "2bin" 추가
        original_filename = secure_filename(file.filename)
        base_name = os.path.splitext(original_filename)[0]  # 확장자 제거
        download_filename = f'{base_name}2bin'
        
        # BIN 파일은 세션에 저장 (다운로드용)
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f'converted_{os.getpid()}.bin')
        os.rename(temp_output.name, session_file)
        
        # 파일명 정보 저장 (다운로드용)
        filename_file = os.path.join(app.config['UPLOAD_FOLDER'], f'filename_{os.getpid()}.txt')
        with open(filename_file, 'w') as f:
            f.write(download_filename)
        
        return jsonify({
            'success': True,
            'preview': f'data:image/jpeg;base64,{preview_data}',
            'stats': stats,
            'download_url': '/download'
        })
        
    except Exception as e:
        return jsonify({'error': f'Conversion error: {str(e)}'}), 500

@app.route('/preview', methods=['POST'])
def preview():
    """Generate preview for BIN file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # 임시 파일로 저장
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        file.save(temp_input.name)
        temp_input.close()
        
        # 미리보기 생성
        temp_preview = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_preview.close()
        
        success = bin_viewer.validate_and_preview(temp_input.name, temp_preview.name)
        
        if not success:
            return jsonify({'error': 'Invalid BIN file'}), 400
        
        # 미리보기 이미지를 base64로 인코딩
        with open(temp_preview.name, 'rb') as f:
            preview_data = base64.b64encode(f.read()).decode('utf-8')
        
        # 통계 정보
        with open(temp_input.name, 'rb') as f:
            bin_data = f.read()
        
        pixel_data = np.frombuffer(bin_data, dtype=np.uint8)
        from collections import Counter
        counter = Counter(pixel_data)
        
        stats = {
            'file_size': len(bin_data),
            'colors': {}
        }
        
        for i in range(6):
            count = counter.get(i, 0)
            percentage = (count / len(pixel_data)) * 100
            stats['colors'][bin_viewer.COLOR_NAMES[i]] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        # 정리
        os.unlink(temp_input.name)
        os.unlink(temp_preview.name)
        
        return jsonify({
            'success': True,
            'preview': f'data:image/jpeg;base64,{preview_data}',
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({'error': f'Preview generation error: {str(e)}'}), 500

@app.route('/download')
def download():
    """Download converted BIN file"""
    pid = os.getpid()
    session_file = os.path.join(app.config['UPLOAD_FOLDER'], f'converted_{pid}.bin')
    filename_file = os.path.join(app.config['UPLOAD_FOLDER'], f'filename_{pid}.txt')
    
    if not os.path.exists(session_file):
        return jsonify({'error': 'No file to download'}), 404
    
    # 저장된 파일명 읽기
    download_filename = 'output2bin'  # 기본값
    if os.path.exists(filename_file):
        with open(filename_file, 'r') as f:
            download_filename = f.read().strip()
        # 정리
        os.unlink(filename_file)
    
    return send_file(session_file, as_attachment=True, download_name=download_filename)

if __name__ == '__main__':
    # templates 디렉토리 생성
    os.makedirs('templates', exist_ok=True)
    
    # 포트 설정 (환경 변수에서 가져오거나 기본값 8000 사용)
    port = int(os.environ.get('PORT', 8000))
    # 배포 환경에서는 debug=False (환경 변수로 제어 가능)
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("=" * 60)
    print("JPG to BIN 변환기 웹 서버 시작")
    print("=" * 60)
    print()
    print("브라우저에서 다음 주소로 접속하세요:")
    print(f"  http://localhost:{port}")
    print()
    print("종료하려면 Ctrl+C를 누르세요")
    print("=" * 60)
    
    app.run(debug=debug, host='0.0.0.0', port=port)
