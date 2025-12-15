# GitHub에서 바로 배포하기

이 프로젝트를 GitHub에 푸시하면 여러 플랫폼에서 바로 배포할 수 있습니다.

## 방법 1: Render.com (추천 - 가장 간단)

1. [Render.com](https://render.com)에 가입 (GitHub 계정으로 로그인 가능)
2. "New +" → "Web Service" 클릭
3. GitHub 저장소 연결
4. 다음 설정 사용:
   - **Name**: jpg-to-bin-converter
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python web_viewer.py`
5. "Create Web Service" 클릭
6. 배포 완료! (약 2-3분 소요)

Render.com은 무료 티어를 제공하며, 자동으로 HTTPS를 지원합니다.

## 방법 2: Railway.app

1. [Railway.app](https://railway.app)에 가입 (GitHub 계정으로 로그인 가능)
2. "New Project" → "Deploy from GitHub repo" 클릭
3. 저장소 선택
4. Railway가 자동으로 `Procfile`을 감지하여 배포합니다
5. 배포 완료!

Railway.app도 무료 크레딧을 제공합니다.

## 방법 3: Fly.io

1. [Fly.io](https://fly.io)에 가입
2. Fly CLI 설치: `curl -L https://fly.io/install.sh | sh`
3. 로그인: `fly auth login`
4. 앱 생성: `fly launch`
5. 배포: `fly deploy`

## 방법 4: Heroku (유료)

1. [Heroku](https://heroku.com)에 가입
2. Heroku CLI 설치
3. 로그인: `heroku login`
4. 앱 생성: `heroku create your-app-name`
5. 배포: `git push heroku main`

## 로컬 테스트

배포 전에 로컬에서 테스트하려면:

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python web_viewer.py

# 브라우저에서 http://localhost:8000 접속
```

## 환경 변수 설정 (필요시)

일부 플랫폼에서는 환경 변수를 설정해야 할 수 있습니다:

- `PORT`: 서버 포트 (대부분의 플랫폼이 자동 설정)
- `HOST`: 호스트 주소 (보통 `0.0.0.0`)

`web_viewer.py`는 이미 `host='0.0.0.0'`으로 설정되어 있어 대부분의 플랫폼에서 작동합니다.

## 포트 설정 (Render.com 등)

일부 플랫폼에서는 환경 변수로 포트를 받아야 합니다. 필요시 `web_viewer.py`를 다음과 같이 수정할 수 있습니다:

```python
import os
port = int(os.environ.get('PORT', 8000))
app.run(debug=False, host='0.0.0.0', port=port)
```

