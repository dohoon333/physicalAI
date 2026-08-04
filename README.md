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
---

## 🍅 방울토마토 수확 전처리 파이프라인 (`harvest_pipeline`)

`docs/`의 PRD·현장조사 체크리스트·전처리 파이프라인 문서를 코드로 구현한 패키지입니다.
CrewAI 앱과는 독립적으로 동작하며, Python 버전 제약도 다릅니다(아래 표 참고).

### 환경 요구사항

| 구성 요소 | Python | 의존성 파일 |
| :--- | :--- | :--- |
| CrewAI 보고서 앱 (`app.py`) | 3.12 (Docker) | `requirements.txt` |
| 전처리 파이프라인 (`harvest_pipeline`) | 3.11 ~ 3.14 | `requirements-pipeline.txt` |

> `crewai`는 Python <3.14 제약이 있어 두 환경을 분리했습니다. 전처리 파이프라인만 사용할 경우
> `requirements-pipeline.txt`만 설치하면 됩니다.

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r requirements-pipeline.txt
```

### 실행

**1) 합성 데이터 데모 — 카메라·데이터 없이 즉시 확인**

```bash
python examples/run_pipeline_demo.py --config configs/default_pipeline.yaml
```

**2) 실제 촬영 파일로 실행**

카메라 내부 파라미터(fx, fy, cx, cy)는 필수입니다. RealSense는
`pyrealsense2`의 `get_intrinsics()`로, 그 외 센서는 체커보드 캘리브레이션으로 구합니다.

```bash
# 단일 프레임
python examples/run_on_images.py \
    --rgb data/truss01.png --depth data/truss01_depth.png \
    --fx 615 --fy 615 --cx 320 --cy 240

# 폴더 일괄 처리 (<name>.png + <name>_depth.png 쌍을 자동 탐색)
python examples/run_on_images.py --input-dir data/ \
    --fx 615 --fy 615 --cx 320 --cy 240 --output results.json
```

지원 입력 형식:

| 입력 | 형식 |
| :--- | :--- |
| RGB | `.png` / `.jpg` / `.bmp` (8bit 3채널) |
| Depth | `.png` (16bit, 값 = mm) 또는 `.npy` (float/int, 값 = mm) |

Depth가 mm 단위가 아니면 `--depth-scale`로 보정합니다(예: 값이 0.1mm 단위면 `--depth-scale 0.1`).
**Depth를 컬러맵으로 저장한 3채널 이미지는 쓸 수 없습니다** — raw 16bit로 저장해야 합니다.

**3) 코드에서 직접 호출**

```python
from harvest_pipeline.config import PipelineConfig
from harvest_pipeline.interfaces import (
    ClassicalColorSegmentationModel, GeometricPedicelPoseEstimator, RuleBasedRipenessClassifier,
)
from harvest_pipeline.pipeline import CameraIntrinsics, FrameInput, HarvestPreprocessingPipeline

with HarvestPreprocessingPipeline(
    config=PipelineConfig.from_yaml("configs/default_pipeline.yaml"),
    segmentation_model=ClassicalColorSegmentationModel(),
    ripeness_classifier=RuleBasedRipenessClassifier(),
    pedicel_estimator=GeometricPedicelPoseEstimator(),
) as pipeline:
    result = pipeline.process_truss(FrameInput(
        rgb=rgb,                    # (H, W, 3) uint8
        depth_mm=depth,             # (H, W) float, 단위 mm
        intrinsics=CameraIntrinsics(fx=615, fy=615, cx=320, cy=240),
        rgb_timestamp_ms=ts_rgb,    # 실시간 스트림에서는 실제 타임스탬프를 넣어야
        depth_timestamp_ms=ts_depth,#   센서 동기화 검사가 의미를 갖는다
        truss_id="TRUSS-01",
    ))

    for fruit in result.harvestable:
        pose = fruit.pose_result.pose
        print(pose.grasp_position_mm, pose.cut_position_mm, pose.rotation_matrix)

    print(result.kpi_summary())
```

**테스트**

```bash
pytest tests/ -v
```

### 실행 결과 확인

- 콘솔: 과실별 판정과 트러스 KPI 요약
- `logs/harvest_pipeline.jsonl`: 스테이지별 소요시간·판정 근거가 담긴 구조화 로그(KPI 집계용)
- `--output results.json`: 프레임별 KPI 요약

### 파이프라인 구조

```
[Stage 0: 공통 전처리]
센서 동기화 → 색상 캘리브레이션 → 하이라이트/그림자 제거 → 구조물 마스킹
  → Depth 필터링 → 인스턴스 세그멘테이션(과실 본체 / Pedicel 마스크 분리)
        ↓
[Stage A: 숙성도 판별 게이트]  ← 여기를 통과한 과실만 Stage B로 진행
HSV/Lab 변환 → 부위별 Multi-patch 샘플링·다수결 → 열과/기형과 예외 필터
        ↓ 신뢰도 ≥90% & 라이트레드~레드
[Stage B: 파지·절단 결합 포즈]
구 피팅(과실 중심) → Pedicel 축 추정 → 고정 오프셋 적용 → 단일 6-DOF 접근 포즈
  → 충돌 검사 → 로봇 실행
```

**Stage A가 Stage B의 하드 게이트입니다.** 숙성도 미달 과실에 파지 계획을 세우는 것은
Edge 하드웨어 연산 낭비이므로 조기 종료시킵니다.

**파지점과 절단점은 별도로 추정하지 않습니다.** 그리퍼와 절단날은 하나의 강체이므로 상대
오프셋은 하드웨어 상수이고, 추정 대상은 "과실 중심 + Pedicel 축" 두 가지뿐입니다.

### 모듈 구성

| 파일 | 역할 |
| :--- | :--- |
| `config.py` | 모든 임계값의 단일 출처(Pydantic 스키마 + YAML 로더) |
| `exceptions.py` | 예외 계층 — 정상 스킵은 예외가 아니라 결과 객체로 표현 |
| `logging_utils.py` | 콘솔 + JSON Lines 이중 로깅, 스테이지별 소요시간 측정 |
| `interfaces.py` | 교체 가능한 AI 모델 Protocol + 규칙 기반 베이스라인 구현 |
| `stage0_common.py` | Stage 0 공통 전처리 함수들 |
| `stage_a_ripeness.py` | Stage A 숙성도 게이트 |
| `stage_b_grasp_cut.py` | Stage B 파지·절단 결합 포즈 |
| `pipeline.py` | 전체 오케스트레이션, 과실 단위 예외 격리, 타임아웃, KPI 집계 |

### 학습된 모델로 교체하기

`interfaces.py`의 세 Protocol을 구현한 클래스를 생성자에 주입하면 됩니다(파이프라인 코드 수정 불필요).

```python
pipeline = HarvestPreprocessingPipeline(
    config=PipelineConfig.from_yaml("configs/greenhouse_A.yaml"),
    segmentation_model=MyYoloV8SegModel(),      # InstanceSegmentationModel
    ripeness_classifier=MyRipenessCNN(),        # RipenessClassifierModel
    pedicel_estimator=MyKeypointRegressor(),    # PedicelPoseEstimator
)
```

기본 제공되는 규칙 기반 구현체(`ClassicalColorSegmentationModel` 등)는 학습 모델이 준비되기
전 파이프라인 검증용이며, 조명 변화가 큰 실제 온실에서는 정확도가 제한적입니다.

### 현장 조사 결과 반영

`configs/default_pipeline.yaml`을 복사해 현장별 설정을 만들고 임계값만 조정합니다.
**코드는 수정하지 않습니다** — 모든 파라미터는 YAML에서만 조정하도록 설계되어 있습니다.

```bash
cp configs/default_pipeline.yaml configs/greenhouse_A.yaml
python examples/run_pipeline_demo.py --config configs/greenhouse_A.yaml
```
