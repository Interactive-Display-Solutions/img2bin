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
import epdoptimize_dither
import image_adjustments

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# 업로드된 이미지 캐시 (세션별)
uploaded_images = {}  # {session_id: {'img_array': ..., 'base_name': ...}}

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
        algorithm = request.form.get('algorithm', 'eink_default')  # 기본값: eink_default (E Ink 공식 알고리즘)
        
        # 임시 파일로 저장
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file.save(temp_input.name)
        temp_input.close()
        
        # 이미지 로드 및 리사이즈
        img = Image.open(temp_input.name).convert('RGB')
        img.thumbnail((1200, 1600), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (1200, 1600), (255, 255, 255))
        offset_x = (1200 - img.width) // 2
        offset_y = (1600 - img.height) // 2
        canvas.paste(img, (offset_x, offset_y))
        img_array = np.array(canvas, dtype=np.uint8)
        
        # BIN 파일 생성
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        temp_output.close()
        
        # 알고리즘에 따라 변환 실행
        if not use_dither:
            result = epdoptimize_dither.quantization_only(img_array)
        elif algorithm == 'eink_default':
            result = jpg_to_bin.eink_default_dither(img_array)
        elif algorithm == 'vivid':
            result = jpg_to_bin.spectra6_vivid_dither(img_array)
        elif algorithm == 'floydSteinberg':
            result = epdoptimize_dither.error_diffusion_dither(img_array, 'floydSteinberg')
        elif algorithm == 'floydSteinberg_serpentine':
            result = epdoptimize_dither.error_diffusion_dither(img_array, 'floydSteinberg', serpentine=True)
        elif algorithm == 'jarvis':
            result = epdoptimize_dither.error_diffusion_dither(img_array, 'jarvis')
        elif algorithm == 'stucki':
            result = epdoptimize_dither.error_diffusion_dither(img_array, 'stucki')
        elif algorithm == 'burkes':
            result = epdoptimize_dither.error_diffusion_dither(img_array, 'burkes')
        elif algorithm == 'sierra3':
            result = epdoptimize_dither.error_diffusion_dither(img_array, 'sierra3')
        elif algorithm == 'sierra2':
            result = epdoptimize_dither.error_diffusion_dither(img_array, 'sierra2')
        elif algorithm == 'sierra2-4a':
            result = epdoptimize_dither.error_diffusion_dither(img_array, 'sierra2-4a')
        elif algorithm == 'bayer4':
            result = epdoptimize_dither.bayer_ordered_dither(img_array, 4)
        elif algorithm == 'bayer8':
            result = epdoptimize_dither.bayer_ordered_dither(img_array, 8)
        elif algorithm == 'ordered':
            result = jpg_to_bin.ordered_dither(img_array)
        else:
            result = jpg_to_bin.eink_default_dither(img_array)
        
        # BIN 파일로 저장
        with open(temp_output.name, 'wb') as f:
            f.write(result.tobytes())
        
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
        
        # 세션에 이미지 저장 (실시간 조정용)
        session_id = str(os.getpid())
        uploaded_images[session_id] = {
            'img_array': img_array.copy(),
            'base_name': base_name
        }
        
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
            'download_url': '/download',
            'session_id': session_id
        })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Conversion error: {error_detail}", flush=True)
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

@app.route('/compare-all', methods=['POST'])
def compare_all():
    """Upload image and compare with bin file in same folder using all algorithms"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file format'}), 400
        # 원본 파일 경로 정보 저장
        original_filename = secure_filename(file.filename)
        base_name = os.path.splitext(original_filename)[0]
        
        # 임시 파일로 저장
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file.save(temp_input.name)
        temp_input.close()
        
        # 이미지 로드 및 리사이즈
        img = Image.open(temp_input.name).convert('RGB')
        img.thumbnail((1200, 1600), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (1200, 1600), (255, 255, 255))
        offset_x = (1200 - img.width) // 2
        offset_y = (1600 - img.height) // 2
        canvas.paste(img, (offset_x, offset_y))
        img_array = np.array(canvas, dtype=np.uint8)
        
        # 모든 알고리즘 정의
        algorithms = {
            'vivid': lambda img: jpg_to_bin.spectra6_vivid_dither(img),
            'floydSteinberg': lambda img: epdoptimize_dither.error_diffusion_dither(img, 'floydSteinberg'),
            'floydSteinberg_serpentine': lambda img: epdoptimize_dither.error_diffusion_dither(img, 'floydSteinberg', serpentine=True),
            'jarvis': lambda img: epdoptimize_dither.error_diffusion_dither(img, 'jarvis'),
            'stucki': lambda img: epdoptimize_dither.error_diffusion_dither(img, 'stucki'),
            'burkes': lambda img: epdoptimize_dither.error_diffusion_dither(img, 'burkes'),
            'sierra3': lambda img: epdoptimize_dither.error_diffusion_dither(img, 'sierra3'),
            'sierra2': lambda img: epdoptimize_dither.error_diffusion_dither(img, 'sierra2'),
            'sierra2-4a': lambda img: epdoptimize_dither.error_diffusion_dither(img, 'sierra2-4a'),
            'bayer4': lambda img: epdoptimize_dither.bayer_ordered_dither(img, 4),
            'bayer8': lambda img: epdoptimize_dither.bayer_ordered_dither(img, 8),
            'ordered': lambda img: jpg_to_bin.ordered_dither(img),
            'quantization_only': epdoptimize_dither.quantization_only,
        }
        
        # 업로드된 파일이 있는 폴더 찾기 (현재 작업 디렉토리 기준)
        # 사용자가 폴더 경로를 제공하거나, 현재 디렉토리에서 검색
        search_folders = ['.', 'test']  # 현재 디렉토리와 test 폴더 검색
        
        reference_bin = None
        reference_bin_path = None
        
        # bin 파일 찾기 (여러 가능한 이름 시도)
        possible_bin_names = [
            f'{base_name}bin',
            f'{base_name}.bin',
            f'{base_name}_reference.bin',
            f'{base_name}_expected.bin',
        ]
        
        for folder in search_folders:
            for bin_name in possible_bin_names:
                bin_path = os.path.join(folder, bin_name)
                if os.path.exists(bin_path):
                    reference_bin_path = bin_path
                    with open(bin_path, 'rb') as f:
                        reference_bin = np.frombuffer(f.read(), dtype=np.uint8)
                    break
            if reference_bin is not None:
                break
        
        results = []
        
        # 각 알고리즘으로 변환 및 비교
        for alg_name, alg_func in algorithms.items():
            temp_preview = None
            temp_bin = None
            try:
                # 변환 실행
                result = alg_func(img_array)
                
                # 미리보기 생성
                temp_preview = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_preview.close()
                
                # 임시 BIN 파일 저장
                temp_bin = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
                temp_bin.write(result.tobytes())
                temp_bin.close()
                
                bin_viewer.validate_and_preview(temp_bin.name, temp_preview.name)
                
                # 미리보기 이미지를 base64로 인코딩
                with open(temp_preview.name, 'rb') as f:
                    preview_data = base64.b64encode(f.read()).decode('utf-8')
                
                # 통계 정보
                pixel_data = result.flatten()
                from collections import Counter
                counter = Counter(pixel_data)
                
                stats = {
                    'file_size': len(pixel_data),
                    'colors': {}
                }
                
                for i in range(6):
                    count = counter.get(i, 0)
                    percentage = (count / len(pixel_data)) * 100
                    stats['colors'][bin_viewer.COLOR_NAMES[i]] = {
                        'count': count,
                        'percentage': round(percentage, 2)
                    }
                
                # 참조 bin 파일과 비교
                error_rate = None
                if reference_bin is not None:
                    if len(result.flatten()) == len(reference_bin):
                        diff = np.sum(result.flatten() != reference_bin)
                        error_rate = (diff / len(reference_bin)) * 100.0
                
                results.append({
                    'algorithm': alg_name,
                    'preview': f'data:image/jpeg;base64,{preview_data}',
                    'stats': stats,
                    'error_rate': round(error_rate, 2) if error_rate is not None else None,
                    'has_reference': reference_bin is not None
                })
                
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"Error in algorithm {alg_name}: {error_detail}", flush=True)
                results.append({
                    'algorithm': alg_name,
                    'error': str(e),
                    'error_rate': None
                })
            finally:
                # 정리
                if temp_preview and os.path.exists(temp_preview.name):
                    try:
                        os.unlink(temp_preview.name)
                    except:
                        pass
                if temp_bin and os.path.exists(temp_bin.name):
                    try:
                        os.unlink(temp_bin.name)
                    except:
                        pass
        
        # 정리
        if os.path.exists(temp_input.name):
            try:
                os.unlink(temp_input.name)
            except:
                pass
        
        # 결과 정렬 (오류율 기준)
        if reference_bin is not None:
            results.sort(key=lambda x: x.get('error_rate', 999) if x.get('error_rate') is not None else 999)
        
        return jsonify({
            'success': True,
            'results': results,
            'reference_found': reference_bin is not None,
            'reference_path': reference_bin_path if reference_bin_path else None
        })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Compare-all error: {error_detail}", flush=True)
        return jsonify({'error': f'Comparison error: {str(e)}'}), 500

@app.route('/compare-single', methods=['POST'])
def compare_single():
    """Single algorithm comparison - returns result immediately"""
    temp_input = None
    temp_preview = None
    temp_bin = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file format'}), 400
        
        algorithm = request.form.get('algorithm', 'floydSteinberg_serpentine')
        
        # 이미지 조정 파라미터
        contrast = float(request.form.get('contrast', 1.0))
        brightness = float(request.form.get('brightness', 1.0))
        saturation = float(request.form.get('saturation', 1.0))
        hue_shift = float(request.form.get('hue_shift', 0.0))
        smoothing = float(request.form.get('smoothing', 0.0))
        
        # 원본 파일 경로 정보 저장
        original_filename = secure_filename(file.filename)
        base_name = os.path.splitext(original_filename)[0]
        
        # 파일을 메모리에 먼저 읽기 (파일 포인터 문제 방지)
        file.seek(0)  # 파일 포인터를 처음으로
        file_data = file.read()
        
        # 임시 파일로 저장
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_input.write(file_data)
        temp_input.close()
        
        # 이미지 로드 및 리사이즈
        img = Image.open(temp_input.name).convert('RGB')
        img.thumbnail((1200, 1600), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (1200, 1600), (255, 255, 255))
        offset_x = (1200 - img.width) // 2
        offset_y = (1600 - img.height) // 2
        canvas.paste(img, (offset_x, offset_y))
        img_array = np.array(canvas, dtype=np.uint8)
        
        # 이미지 조정 파라미터 추가
        smoothing = float(request.form.get('smoothing', 0.0))
        
        # 이미지 조정 적용
        if contrast != 1.0 or brightness != 1.0 or saturation != 1.0 or hue_shift != 0.0 or smoothing > 0.0:
            img_array = image_adjustments.adjust_image(
                img_array,
                contrast=contrast,
                brightness=brightness,
                saturation=saturation,
                hue_shift=hue_shift,
                smoothing=smoothing
            )
        
        
        # 참조 bin 파일 찾기
        search_folders = ['.', 'test']
        reference_bin = None
        reference_bin_path = None
        
        possible_bin_names = [
            f'{base_name}bin',
            f'{base_name}.bin',
            f'{base_name}_reference.bin',
            f'{base_name}_expected.bin',
        ]
        
        for folder in search_folders:
            for bin_name in possible_bin_names:
                bin_path = os.path.join(folder, bin_name)
                if os.path.exists(bin_path):
                    reference_bin_path = bin_path
                    with open(bin_path, 'rb') as f:
                        reference_bin = np.frombuffer(f.read(), dtype=np.uint8)
                    break
            if reference_bin is not None:
                break
        
        # 알고리즘 정의 (eink_default 추가)
        if algorithm == 'eink_default':
            alg_func = lambda img: jpg_to_bin.eink_default_dither(img)
        elif algorithm == 'vivid':
            alg_func = lambda img: jpg_to_bin.spectra6_vivid_dither(img)
        elif algorithm == 'floydSteinberg':
            alg_func = lambda img: epdoptimize_dither.error_diffusion_dither(img, 'floydSteinberg')
        elif algorithm == 'floydSteinberg_serpentine':
            alg_func = lambda img: epdoptimize_dither.error_diffusion_dither(img, 'floydSteinberg', serpentine=True)
        elif algorithm == 'jarvis':
            alg_func = lambda img: epdoptimize_dither.error_diffusion_dither(img, 'jarvis')
        elif algorithm == 'stucki':
            alg_func = lambda img: epdoptimize_dither.error_diffusion_dither(img, 'stucki')
        elif algorithm == 'burkes':
            alg_func = lambda img: epdoptimize_dither.error_diffusion_dither(img, 'burkes')
        elif algorithm == 'sierra3':
            alg_func = lambda img: epdoptimize_dither.error_diffusion_dither(img, 'sierra3')
        elif algorithm == 'sierra2':
            alg_func = lambda img: epdoptimize_dither.error_diffusion_dither(img, 'sierra2')
        elif algorithm == 'sierra2-4a':
            alg_func = lambda img: epdoptimize_dither.error_diffusion_dither(img, 'sierra2-4a')
        elif algorithm == 'bayer4':
            alg_func = lambda img: epdoptimize_dither.bayer_ordered_dither(img, 4)
        elif algorithm == 'bayer8':
            alg_func = lambda img: epdoptimize_dither.bayer_ordered_dither(img, 8)
        elif algorithm == 'ordered':
            alg_func = lambda img: jpg_to_bin.ordered_dither(img)
        elif algorithm == 'quantization_only':
            alg_func = epdoptimize_dither.quantization_only
        else:
            alg_func = lambda img: jpg_to_bin.eink_default_dither(img)
        
        # 변환 실행
        try:
            result = alg_func(img_array)
            
            # 미리보기 생성
            temp_preview = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_preview.close()
            
            # 임시 BIN 파일 저장
            temp_bin = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
            temp_bin.write(result.tobytes())
            temp_bin.close()
            
            bin_viewer.validate_and_preview(temp_bin.name, temp_preview.name)
            
            # 미리보기 이미지를 base64로 인코딩
            with open(temp_preview.name, 'rb') as f:
                preview_data = base64.b64encode(f.read()).decode('utf-8')
            
            # 통계 정보
            pixel_data = result.flatten()
            from collections import Counter
            counter = Counter(pixel_data)
            
            stats = {
                'file_size': len(pixel_data),
                'colors': {}
            }
            
            for i in range(6):
                count = counter.get(i, 0)
                percentage = (count / len(pixel_data)) * 100
                stats['colors'][bin_viewer.COLOR_NAMES[i]] = {
                    'count': count,
                    'percentage': round(percentage, 2)
                }
            
            # 참조 bin 파일과 비교
            error_rate = None
            if reference_bin is not None:
                if len(result.flatten()) == len(reference_bin):
                    diff = np.sum(result.flatten() != reference_bin)
                    error_rate = (diff / len(reference_bin)) * 100.0
            
            return jsonify({
                'success': True,
                'algorithm': algorithm,
                'preview': f'data:image/jpeg;base64,{preview_data}',
                'stats': stats,
                'error_rate': round(error_rate, 2) if error_rate is not None else None,
                'has_reference': reference_bin is not None,
                'reference_path': reference_bin_path if reference_bin_path else None
            })
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"Error in algorithm {algorithm}: {error_detail}", flush=True)
            return jsonify({
                'success': False,
                'algorithm': algorithm,
                'error': str(e)
            }), 500
        finally:
            # 정리
            if temp_preview and os.path.exists(temp_preview.name):
                try:
                    os.unlink(temp_preview.name)
                except:
                    pass
            if temp_bin and os.path.exists(temp_bin.name):
                try:
                    os.unlink(temp_bin.name)
                except:
                    pass
            if os.path.exists(temp_input.name):
                try:
                    os.unlink(temp_input.name)
                except:
                    pass
                    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Compare-single error: {error_detail}", flush=True)
        return jsonify({'error': f'Comparison error: {str(e)}'}), 500

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
    
    return send_file(
        session_file,
        as_attachment=True,
        download_name=f'{download_filename}.bin',
        mimetype='application/octet-stream'
    )

@app.route('/download-live')
def download_live():
    """Download live preview BIN file"""
    session_id = request.args.get('session_id', str(os.getpid()))
    pid = os.getpid()
    session_file = os.path.join(app.config['UPLOAD_FOLDER'], f'live_preview_{pid}_{session_id}.bin')
    filename_file = os.path.join(app.config['UPLOAD_FOLDER'], f'live_preview_filename_{pid}_{session_id}.txt')
    
    if not os.path.exists(session_file):
        return jsonify({'error': 'No preview file to download. Please update preview first.'}), 404
    
    # 저장된 파일명 읽기
    download_filename = 'preview_output'  # 기본값
    if os.path.exists(filename_file):
        with open(filename_file, 'r') as f:
            download_filename = f.read().strip()
    
    return send_file(
        session_file,
        as_attachment=True,
        download_name=f'{download_filename}.bin',
        mimetype='application/octet-stream'
    )

@app.route('/preview-live', methods=['POST'])
def preview_live():
    """실시간 미리보기 (이미지 조정 + 디더링)"""
    try:
        session_id = request.form.get('session_id', str(os.getpid()))
        
        if session_id not in uploaded_images:
            # 세션이 없으면 조용히 에러 반환 (이미지 업로드 전에는 정상)
            return jsonify({'error': 'No image uploaded. Please upload an image first.', 'success': False}), 200
        
        # 파라미터 읽기
        algorithm = request.form.get('algorithm', 'floydSteinberg_serpentine')
        contrast = float(request.form.get('contrast', 1.0))
        brightness = float(request.form.get('brightness', 1.0))
        saturation = float(request.form.get('saturation', 1.0))
        hue_shift = float(request.form.get('hue_shift', 0.0))
        smoothing = float(request.form.get('smoothing', 0.0))
        
        # 이미지 가져오기
        cached = uploaded_images[session_id]
        img_array = cached['img_array'].copy()
        
        # 이미지 조정 적용
        adjusted_img = image_adjustments.adjust_image(
            img_array,
            contrast=contrast,
            brightness=brightness,
            saturation=saturation,
            hue_shift=hue_shift,
            smoothing=smoothing
        )
        
        # 디더링 적용
        if algorithm == 'eink_default':
            result = jpg_to_bin.eink_default_dither(adjusted_img)
        elif algorithm == 'vivid':
            result = jpg_to_bin.spectra6_vivid_dither(adjusted_img)
        elif algorithm == 'floydSteinberg':
            result = epdoptimize_dither.error_diffusion_dither(adjusted_img, 'floydSteinberg')
        elif algorithm == 'floydSteinberg_serpentine':
            result = epdoptimize_dither.error_diffusion_dither(adjusted_img, 'floydSteinberg', serpentine=True)
        elif algorithm == 'jarvis':
            result = epdoptimize_dither.error_diffusion_dither(adjusted_img, 'jarvis')
        elif algorithm == 'stucki':
            result = epdoptimize_dither.error_diffusion_dither(adjusted_img, 'stucki')
        elif algorithm == 'burkes':
            result = epdoptimize_dither.error_diffusion_dither(adjusted_img, 'burkes')
        elif algorithm == 'sierra3':
            result = epdoptimize_dither.error_diffusion_dither(adjusted_img, 'sierra3')
        elif algorithm == 'sierra2':
            result = epdoptimize_dither.error_diffusion_dither(adjusted_img, 'sierra2')
        elif algorithm == 'sierra2-4a':
            result = epdoptimize_dither.error_diffusion_dither(adjusted_img, 'sierra2-4a')
        elif algorithm == 'bayer4':
            result = epdoptimize_dither.bayer_ordered_dither(adjusted_img, 4)
        elif algorithm == 'bayer8':
            result = epdoptimize_dither.bayer_ordered_dither(adjusted_img, 8)
        elif algorithm == 'ordered':
            result = jpg_to_bin.ordered_dither(adjusted_img)
        else:
            result = jpg_to_bin.eink_default_dither(adjusted_img)
        
        # 미리보기 생성
        temp_bin = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        temp_bin.write(result.tobytes())
        temp_bin.close()
        
        temp_preview = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_preview.close()
        
        bin_viewer.validate_and_preview(temp_bin.name, temp_preview.name)
        
        # 미리보기 이미지를 base64로 인코딩
        with open(temp_preview.name, 'rb') as f:
            preview_data = base64.b64encode(f.read()).decode('utf-8')
        
        # 통계 정보
        pixel_data = result.flatten()
        from collections import Counter
        counter = Counter(pixel_data)
        
        stats = {
            'file_size': len(pixel_data),
            'colors': {}
        }
        
        for i in range(6):
            count = counter.get(i, 0)
            percentage = (count / len(pixel_data)) * 100
            stats['colors'][bin_viewer.COLOR_NAMES[i]] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        # BIN 파일을 세션에 저장 (다운로드용)
        pid = os.getpid()
        session_bin_file = os.path.join(app.config['UPLOAD_FOLDER'], f'live_preview_{pid}_{session_id}.bin')
        with open(session_bin_file, 'wb') as f:
            f.write(result.tobytes())
        
        # 파일명 저장
        session_filename_file = os.path.join(app.config['UPLOAD_FOLDER'], f'live_preview_filename_{pid}_{session_id}.txt')
        with open(session_filename_file, 'w') as f:
            f.write(f'preview_{algorithm}_{int(contrast*100)}_{int(brightness*100)}_{int(saturation*100)}_{int(hue_shift)}_{int(smoothing*10)}.bin')
        
        # 정리 (임시 파일만 삭제, 세션 파일은 유지)
        if os.path.exists(temp_bin.name):
            try:
                os.unlink(temp_bin.name)
            except:
                pass
        if os.path.exists(temp_preview.name):
            try:
                os.unlink(temp_preview.name)
            except:
                pass
        
        return jsonify({
            'success': True,
            'preview': f'data:image/jpeg;base64,{preview_data}',
            'stats': stats,
            'download_url': f'/download-live?session_id={session_id}'
        })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Live preview error: {error_detail}", flush=True)
        return jsonify({'error': f'Live preview error: {str(e)}'}), 500

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
