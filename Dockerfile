# 1. 파이썬 3.12 슬림 베이스 이미지 사용
FROM python:3.12-slim

# 2. 한글 출력 및 파이썬 버퍼링 방지 환경변수 설정
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스코드 및 필요 폴더 복사
COPY . .

# 6. app.py 대화형(Interactive) 실행
CMD ["python", "app.py"]