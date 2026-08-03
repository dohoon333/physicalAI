# 🤖 Physical AI Multi-Agent System (CrewAI + Gemini 3.1)

본 프로젝트는 **Physical AI 스타트업의 CEO(수강생)**가 되어 자율형 멀티 에이전트 조직에게 업무 지시를 내리고, **총괄 PM의 지휘 아래 필요한 전문 부서들이 선택적으로 호출되어 최종 프로젝트 결과 보고서를 도출하는 시스템**입니다.

Docker 컨테이너 환경 기반으로 구축되어, 별도의 파이썬 환경 설정 없이 명령 한 줄로 실행 가능합니다.

---

## 🏢 조직 구조 (Organizational Chart)

* **CEO (수강생):** 터미널을 통해 프로젝트의 방향성 및 돌발 지시어 입력
* **총괄 PM (Chief PM):** CEO의 지시를 분석하여 불필요한 공수를 줄이고, **필요한 전문 부서에만 선택적으로 업무를 할당(Hierarchical Process)**한 뒤 최종 보고서 작성
* **하위 전문 부서:**
  * 🦾 **하드웨어 & 메카트로닉스 팀:** 기계 구조, 센서 배치, 케이싱 설계
  * ⚡ **임베디드 & 로봇 제어 팀:** MCU 펌웨어, ROS2 제어, 통신 및 모터 제어
  * 👁️ **AI & 데이터 파이프라인 팀:** 컴퓨터 비전, AI 모델 학습, ONNX/TensorRT 최적화
  * 📊 **비즈니스 & 프로덕트 기획 팀:** PRD 작성, 사업성 평가, ROI 산출
  * 🌐 **웹 개발 & MLOps 팀:** FastAPI 백엔드, 프론트엔드 UI, 로그인/인증 체계 및 대시보드

---

## 🛠️ 실습 전 준비 사항 (Prerequisites)

1. **Docker Desktop**이 설치되어 실행 중이어야 합니다. ([Docker Desktop 다운로드](https://www.docker.com/products/docker-desktop/))
2. Google Gemini API 키가 필요합니다. ([Google AI Studio](https://aistudio.google.com/)에서 무료 발급 가능)

---

## 🚀 빠른 시작 및 실행 가이드 (Quick Start)

### 1단계: 환경변수 파일 생성 및 API 키 설정 (`.env`)

1. 프로젝트 폴더 내의 **`.env.example`** 파일의 이름을 **`.env`** 로 변경합니다.
2. `.env` 파일을 열고, 발급받은 Gemini API 키를 입력합니다.

```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

### 2단계: Docker 이미지 빌드

프로젝트에서 사용하는 Docker 이미지를 생성합니다. 최초 실행 시 또는 Dockerfile이 변경되었을 때 실행합니다.

```bash
docker compose build
```

### 3단계: 프로젝트 실행

아래 명령어를 실행하면 Docker 컨테이너가 생성되고, Physical AI Multi-Agent System이 실행됩니다.

```bash
docker compose run --rm physical-ai-app
```

`--rm` 옵션은 실행이 종료되면 컨테이너를 자동으로 삭제하여 불필요한 컨테이너가 남지 않도록 합니다.