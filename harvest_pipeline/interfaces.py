"""교체 가능한 AI 모델 컴포넌트의 인터페이스(Protocol) 정의.

PRD 4장 기술스택이 지정하는 세 가지 모델(인스턴스 세그멘테이션, 숙성도 분류, 파지·절단
결합 포즈용 Pedicel 축 추정)은 실제로는 각각 YOLOv8-seg/FastSAM, CNN 색상 분류기,
커스텀 키포인트 회귀 모델로 교체되어야 한다. 이 모듈은 그 교체 지점을 명확한 Protocol로
고정해, 파이프라인 나머지 코드가 구체적인 모델 구현에 의존하지 않도록(의존성 역전) 한다.

각 Protocol마다 딥러닝 없이도 즉시 동작하는 규칙 기반(Classical CV) 기본 구현체를 함께
제공한다. 이는 (1) 실제 학습된 모델이 준비되기 전에도 파이프라인 전체를 end-to-end로
검증할 수 있게 하고, (2) 학습된 모델 도입 후에도 베이스라인 비교군으로 활용할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from harvest_pipeline.config import RipenessGateConfig, RipenessStage
from harvest_pipeline.exceptions import InvalidImageError, InvalidPointCloudError


def _validate_hwc_image(image: np.ndarray, *, name: str, channels: int = 3) -> None:
    if image.ndim != 3 or image.shape[2] != channels:
        raise InvalidImageError(
            f"{name}는 (H, W, {channels}) 형태의 배열이어야 합니다. 실제 shape={image.shape}"
        )
    if image.size == 0:
        raise InvalidImageError(f"{name}가 비어 있습니다(size=0).")


def _validate_point_cloud(points: np.ndarray, *, name: str = "points") -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise InvalidPointCloudError(
            f"{name}는 (N, 3) 형태의 배열이어야 합니다. 실제 shape={points.shape}"
        )


# ---------------------------------------------------------------------------
# 1. Instance Segmentation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SegmentationInstance:
    """개별 검출 인스턴스 하나(과실 1개 또는 잎/줄기 영역 1개).

    과실 본체 마스크(`mask`)와 Pedicel 마스크(`pedicel_mask`)를 분리해 보관하는 이유:
    - Stage A의 숙성도 판정과 기형과(원형도) 검사는 **과실 본체만** 대상으로 해야 한다.
      Pedicel이 마스크에 섞여 있으면 가늘고 긴 돌출부 때문에 둘레가 길어져 원형도가
      떨어지고, 정상 과실이 기형과로 오판정된다(실제로 이 문제가 확인되었다).
    - 반면 Stage B의 Pedicel 축 추정은 **본체 + Pedicel을 합친** 포인트가 필요하다.
      본체만으로는 상단이 납작한 구면 캡이 되어 축 방향을 잡을 수 없다.
    """

    instance_id: int
    class_label: str  # "fruit" | "leaf" | "stem"
    mask: np.ndarray  # bool, shape (H, W) — 과실 본체만
    bbox_xyxy: tuple[int, int, int, int]  # 과실 본체 기준 bbox
    confidence: float
    pedicel_mask: np.ndarray | None = None  # bool, shape (H, W) — 꼭지 영역(있으면)

    @property
    def combined_mask(self) -> np.ndarray:
        """과실 본체 + Pedicel을 합친 마스크(Stage B 포인트클라우드 추출용)."""
        if self.pedicel_mask is None:
            return self.mask
        return self.mask | self.pedicel_mask


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    instances: tuple[SegmentationInstance, ...] = field(default_factory=tuple)

    @property
    def fruit_instances(self) -> tuple[SegmentationInstance, ...]:
        return tuple(inst for inst in self.instances if inst.class_label == "fruit")


@runtime_checkable
class InstanceSegmentationModel(Protocol):
    """실전 배포 시 YOLOv8-seg/FastSAM으로 교체되는 지점(PRD 4장)."""

    def predict(self, rgb: np.ndarray) -> SegmentationResult:
        """RGB 이미지에서 과실/잎/줄기 인스턴스를 분할한다."""
        ...


class ClassicalColorSegmentationModel:
    """HSV 색상 임계값 + 연결 성분 분석 + 형태학적 연산 기반의 규칙 기반 세그멘테이션.

    학습된 모델이 없는 초기 개발/테스트 단계에서 파이프라인을 즉시 실행 가능하게 하는
    베이스라인이다. 각 연결 성분에 대해 열림(Opening) 연산으로 가늘고 긴 Pedicel을 제거해
    과실 본체를 얻고, 원본에서 본체를 빼서 Pedicel 마스크를 분리한다.

    한계(프로덕션에서 YOLOv8-seg 등 학습 모델로 교체해야 하는 이유, PRD 4장):
    - 색상 임계값이 고정되어 있어 조명이 크게 변하는 실제 온실에서는 취약하다.
    - 잎과 색이 겹치는 미숙과(Green 단계)는 원리적으로 분리할 수 없다. 따라서 이 베이스라인
      으로는 "미숙과를 탐지한 뒤 게이트에서 걸러내는" 흐름을 완전히 재현할 수 없고,
      브레이커 단계 이후(색이 돌기 시작한) 과실만 탐지된다.
    """

    def __init__(
        self,
        min_instance_area_px: int = 150,
        pedicel_removal_kernel_px: int = 11,
    ) -> None:
        """
        Args:
            min_instance_area_px: 이 면적 미만의 연결 성분은 노이즈로 간주해 버린다.
            pedicel_removal_kernel_px: 열림 연산 커널 지름. Pedicel 두께보다 크고 과실
                지름보다 작아야 한다 — 이 조건이 깨지면 Pedicel이 남거나(너무 작을 때)
                과실 본체까지 침식된다(너무 클 때).
        """
        self._min_instance_area_px = min_instance_area_px
        self._pedicel_removal_kernel_px = pedicel_removal_kernel_px

    def predict(self, rgb: np.ndarray) -> SegmentationResult:
        _validate_hwc_image(rgb, name="rgb")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # 색이 돌기 시작한 과실(브레이커~레드)은 Hue가 0~35 또는 165~180 부근에 위치한다
        # (OpenCV Hue는 0~179 스케일). 잎의 녹색(약 35~85)은 제외된다.
        warm_low = cv2.inRange(hsv, (0, 80, 60), (35, 255, 255))
        warm_high = cv2.inRange(hsv, (165, 80, 60), (180, 255, 255))
        fruit_region = cv2.bitwise_or(warm_low, warm_high)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fruit_region, connectivity=8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._pedicel_removal_kernel_px, self._pedicel_removal_kernel_px),
        )

        instances: list[SegmentationInstance] = []
        for label_id in range(1, num_labels):  # 0번은 배경
            if int(stats[label_id, cv2.CC_STAT_AREA]) < self._min_instance_area_px:
                continue

            component = (labels == label_id).astype(np.uint8)
            # 열림 연산: 침식 후 팽창 → 커널보다 얇은 구조(Pedicel)는 사라지고 본체만 남는다.
            body = cv2.morphologyEx(component, cv2.MORPH_OPEN, kernel)
            if int(np.count_nonzero(body)) < self._min_instance_area_px:
                # 열림 후 본체가 남지 않으면 과실이 아니라 얇은 줄기/노이즈로 판단.
                continue

            body_mask = body.astype(bool)
            pedicel_mask = component.astype(bool) & ~body_mask

            ys, xs = np.where(body_mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

            instances.append(
                SegmentationInstance(
                    instance_id=label_id,
                    class_label="fruit",
                    mask=body_mask,
                    bbox_xyxy=bbox,
                    confidence=1.0,  # 규칙 기반이라 확률적 신뢰도가 없어 상한값으로 고정
                    pedicel_mask=pedicel_mask if np.any(pedicel_mask) else None,
                )
            )
        return SegmentationResult(instances=tuple(instances))


# ---------------------------------------------------------------------------
# 2. Ripeness Classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RipenessPrediction:
    stage: RipenessStage
    confidence: float
    patch_votes: tuple[RipenessStage, ...] = field(default_factory=tuple)


@runtime_checkable
class RipenessClassifierModel(Protocol):
    """실전 배포 시 ResNet/EfficientNet 기반 색상 분류 CNN으로 교체되는 지점(PRD 4장)."""

    def predict(self, patches_rgb: list[np.ndarray]) -> RipenessPrediction:
        """과실 표면 여러 부위(Multi-patch)의 RGB 패치로부터 숙성 단계를 판정한다."""
        ...


# 표준 6단계의 평균 Hue 기준값(대략치). 실제 배포 시 품종별 캘리브레이션 테이블로 대체된다
# (체크리스트 A-1, 전처리 문서 1장 "품종별 색상 기준값 캘리브레이션 테이블").
_DEFAULT_STAGE_HUE_CENTERS: dict[RipenessStage, float] = {
    RipenessStage.GREEN: 55.0,
    RipenessStage.BREAKER: 40.0,
    RipenessStage.TURNING: 25.0,
    RipenessStage.PINK: 10.0,
    RipenessStage.LIGHT_RED: 5.0,
    RipenessStage.RED: 0.0,
}


def _circular_mean_hue_degrees(hue_raw_0_179: np.ndarray) -> float:
    """OpenCV Hue(0~179 스케일)를 0~360도 원형(circular) 평균으로 환산한다.

    Hue는 색상환(color wheel) 위의 각도이므로 선형 평균을 취하면 안 된다. 예를 들어
    순수 적색을 나타내는 OpenCV Hue=2와 Hue=178(둘 다 0/360도 부근)을 산술 평균하면
    90(=180도, 청록색 부근)이 되어 버려 완전히 엉뚱한 색으로 오염된다. 이를 막기 위해
    각 Hue를 단위원 위의 벡터 (cos θ, sin θ)로 변환해 벡터 평균을 낸 뒤 각도로 되돌리는
    원형 평균(순환 통계학의 표준 기법)을 사용한다.

    변환 지점: 입력은 OpenCV 스케일(0~179)이므로 먼저 *2를 해 실제 각도(0~360도)로
    바꾼 다음 라디안으로 변환해 삼각함수에 넣는다.
    """
    hue_degrees = hue_raw_0_179.astype(np.float64) * 2.0
    theta = np.deg2rad(hue_degrees)
    mean_sin = float(np.mean(np.sin(theta)))
    mean_cos = float(np.mean(np.cos(theta)))
    mean_angle_degrees = float(np.degrees(np.arctan2(mean_sin, mean_cos)))
    return mean_angle_degrees % 360.0


def _circular_hue_distance_degrees(a_degrees: float, b_degrees: float) -> float:
    """색상환 위 두 각도(0~360도) 사이의 최단 순환 거리.

    예: 350도와 0도는 선형 거리로는 350이지만 색상환에서는 10만큼만 떨어져 있다(둘 다
    붉은색 부근). 이 거리 함수를 쓰지 않으면 Hue가 350도인 붉은 과실이 green(55도)보다
    red(0도)에서 더 멀다고 잘못 판단해 green으로 오분류된다(실측 확인된 결함).
    """
    diff = abs(a_degrees - b_degrees) % 360.0
    return min(diff, 360.0 - diff)


class RuleBasedRipenessClassifier:
    """부위별(Multi-patch) 평균 Hue를 최근접 숙성 단계 기준값에 매칭하고 다수결로 판정한다.

    전처리 문서 1장의 "Fruit 부위별 Multi-patch 색상 샘플링 + 다수결 판정" 기법을 그대로
    구현한 규칙 기반 베이스라인이다. 프로덕션에서는 CNN 분류기로 교체하되, 이 다수결 로직
    자체(단일 평균색 대신 부위별 판정 후 합산)는 유지하는 것을 권장한다.

    신뢰도(confidence)는 두 요소를 곱해서 산출한다:
    - vote_ratio: 다수결에 참여한 패치 중 다수 의견이 차지하는 비율. 패치 수가 달라지면
      가능한 비율값 자체는 여전히 이산적이지만(예: 4/5=0.8, 4/4=1.0),
    - angular_score: 다수 의견에 투표한 패치들이 실제로 그 단계 기준 Hue에 얼마나
      가까웠는지를 나타내는 연속값(1에 가까울수록 기준값과 거의 일치).

    이 둘을 곱하는 이유: vote_ratio만 쓰면(수정 전 구현) 패치 수가 줄어들수록 만장일치를
    달성하기 쉬워져 confidence=1.0에 오히려 유리해지는 역전 현상이 생긴다(패치 4개 중
    4개 일치와 패치 5개 중 4개 일치는 판정의 확실성이 다른데 전자만 1.0이 됨). 반면
    angular_score는 패치 수와 무관하게 "색이 기준값에서 얼마나 벗어났는가"만 반영하므로,
    설령 우연히 만장일치가 나오더라도 실제 색이 기준에서 멀면 confidence가 낮게 유지된다.
    """

    def __init__(
        self,
        stage_hue_centers: dict[RipenessStage, float] | None = None,
        hue_tolerance_degrees: float = 45.0,
    ) -> None:
        """
        Args:
            stage_hue_centers: 숙성 단계별 기준 Hue(도 단위, 0~360). 기본값은
                `_DEFAULT_STAGE_HUE_CENTERS`(품종별 캘리브레이션 이전의 대략치).
            hue_tolerance_degrees: angular_score 계산 시 "이 거리 이상 벗어나면 신뢰도
                기여가 0"으로 보는 허용 오차(도 단위). 기준값 간 최대 간격(약 15도)의 3배
                수준으로 넉넉히 잡았다 — `_DEFAULT_STAGE_HUE_CENTERS`는 "품종별 캘리브레이션
                이전의 대략치"일 뿐이라 실제 과실 색이 기준값에서 몇 도 어긋나는 것은
                정상이며(예: breaker 기준 40도인데 실측 36도), 이 정도 편차로 만장일치 판정이
                confidence_threshold(기본 0.90) 문턱을 넘지 못하면 정상 과실이 전부 재촬영·
                폐기 처리된다. 반면 진짜로 서로 다른 색이 섞인 경우(좌우 절반이 다른 색 등)는
                거리가 이 허용치를 훨씬 초과하므로 여전히 낮은 confidence로 걸러진다.
                config.py 수정 없이 조정할 수 있도록 생성자 파라미터로 노출한다.
        """
        self._stage_hue_centers = stage_hue_centers or _DEFAULT_STAGE_HUE_CENTERS
        self._hue_tolerance_degrees = hue_tolerance_degrees

    def predict(self, patches_rgb: list[np.ndarray]) -> RipenessPrediction:
        if not patches_rgb:
            raise InvalidImageError("숙성도 판정을 위한 패치가 1개 이상 필요합니다.")

        votes: list[RipenessStage] = []
        nearest_distances: list[float] = []
        for patch in patches_rgb:
            _validate_hwc_image(patch, name="ripeness patch")
            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            mean_hue_degrees = _circular_mean_hue_degrees(hsv[:, :, 0])
            nearest_stage = min(
                self._stage_hue_centers,
                key=lambda stage: _circular_hue_distance_degrees(
                    self._stage_hue_centers[stage], mean_hue_degrees
                ),
            )
            votes.append(nearest_stage)
            nearest_distances.append(
                _circular_hue_distance_degrees(
                    self._stage_hue_centers[nearest_stage], mean_hue_degrees
                )
            )

        majority_stage = max(set(votes), key=votes.count)
        vote_ratio = votes.count(majority_stage) / len(votes)

        majority_distances = [
            distance
            for stage, distance in zip(votes, nearest_distances)
            if stage == majority_stage
        ]
        mean_majority_distance = float(np.mean(majority_distances))
        angular_score = float(
            np.clip(1.0 - mean_majority_distance / self._hue_tolerance_degrees, 0.0, 1.0)
        )
        confidence = vote_ratio * angular_score

        return RipenessPrediction(stage=majority_stage, confidence=confidence, patch_votes=tuple(votes))


def sample_multi_patch_regions(
    rgb: np.ndarray, mask: np.ndarray, config: RipenessGateConfig
) -> list[np.ndarray]:
    """과실 마스크 내부를 2D 격자로 등분해 부위별 색상 패치를 추출한다(불균일 발색 대응, 체크리스트 A-3).

    마스크의 bounding box를 세로·가로 양방향으로 나눈 격자 셀마다 **마스크에 포함된 픽셀만**
    모아 (N, 1, 3) 형태의 패치로 반환한다. `config.multi_patch_sample_count`(설정 파일 값,
    변경 불가)는 "총 패치 수의 목표치"로 해석해 열(cols) x 행(rows) 격자로 분배한다. 열 수를
    ceil(sqrt(count))로 잡고 행 수를 나머지 몫으로 채우면(예: count=5 -> 2행 3열) 정사각형에
    가까운 격자가 되어 세로·가로 어느 방향의 불균일 발색도 고르게 가로지른다.

    세로 방향으로만 등분하던 이전 구현의 결함: 좌우로 절반씩 다른 색(예: 좌측은 완숙,
    우측은 미숙)인 과실에서 각 가로 스트라이프 내부에 좌우 색이 모두 섞여 평균 Hue가
    중간값으로 뭉개진다. 더 나쁜 것은, 스트라이프 방향이 우연히 좌우 색 경계와 평행하면
    각 스트라이프가 한쪽 색으로만 채워져 다수결이 만장일치(confidence=1.0)로 나오는데,
    이는 "불균일 발색을 포착하지 못한 것"에 그치지 않고 저신뢰도 안전망(재촬영 유도)까지
    무력화하는 문제였다(실제로 좌우 분할 케이스에서 확인됨). 2D 격자는 세로 스트라이프와
    가로 스트라이프를 모두 포함하므로 어느 방향의 경계든 최소 한 축에서는 서로 다른 셀로
    분리되어 다수결에 반영된다.

    bounding box를 직사각형으로 통째로 잘라내지 않는 이유(격자 셀 자체에도 동일하게 적용):
    과실은 구형이라 bbox 모서리에는 잎이나 배경 픽셀이 섞여 들어가고, 그 픽셀들이 평균
    Hue를 오염시켜 숙성 단계를 오판별하게 만든다(실제로 이 방식에서 완전히 붉은 과실이
    light_red로 잘못 분류되는 문제가 확인되었다). 따라서 각 셀에서도 마스크 픽셀만 골라낸다.
    반환 형태가 (N, 1, 3)인 것은 cv2.cvtColor 등 OpenCV 함수가 요구하는 HWC 레이아웃을
    유지하면서 순수 과실 픽셀만 담기 위한 것이다.

    극단적으로 작은 마스크나 격자 셀에 마스크 픽셀이 하나도 없는 경우 유효 패치가
    sample_count보다 적을 수 있으며, 이는 정상 동작이다(호출부에서 개수를 확인한다).
    """
    _validate_hwc_image(rgb, name="rgb")
    if mask.shape != rgb.shape[:2]:
        raise InvalidImageError(
            f"mask shape {mask.shape}가 rgb shape {rgb.shape[:2]}와 일치하지 않습니다."
        )

    ys, xs = np.where(mask)
    if ys.size == 0:
        return []

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    cols = max(1, int(np.ceil(np.sqrt(config.multi_patch_sample_count))))
    rows = max(1, int(np.ceil(config.multi_patch_sample_count / cols)))

    y_edges = np.linspace(y_min, y_max + 1, rows + 1, dtype=int)
    x_edges = np.linspace(x_min, x_max + 1, cols + 1, dtype=int)

    patches: list[np.ndarray] = []
    for y_start, y_end in zip(y_edges[:-1], y_edges[1:]):
        row_mask = (ys >= y_start) & (ys < y_end)
        if not np.any(row_mask):
            continue
        for x_start, x_end in zip(x_edges[:-1], x_edges[1:]):
            cell_mask = row_mask & (xs >= x_start) & (xs < x_end)
            if not np.any(cell_mask):
                continue
            cell_pixels = rgb[ys[cell_mask], xs[cell_mask]]  # shape (N, 3), 마스크 픽셀만
            patches.append(cell_pixels.reshape(-1, 1, 3))
    return patches


# ---------------------------------------------------------------------------
# 3. Pedicel Axis Estimation (Stage B 결합 포즈의 입력)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PedicelAxisEstimate:
    origin_mm: np.ndarray  # shape (3,)
    direction_unit: np.ndarray  # shape (3,), 단위 벡터
    confidence: float  # PCA 설명 분산비 등 추정 신뢰도(0~1)


@runtime_checkable
class PedicelPoseEstimator(Protocol):
    """실전 배포 시 커스텀 키포인트 회귀 모델로 교체되는 지점(PRD 4장)."""

    def estimate(self, points_above_fruit_mm: np.ndarray) -> PedicelAxisEstimate:
        """과실 중심 상단부 포인트클라우드로부터 Pedicel 축 방향을 추정한다."""
        ...


class GeometricPedicelPoseEstimator:
    """PCA(주성분분석) 기반 축 방향 추정 — 딥러닝 없이도 동작하는 기하학적 베이스라인.

    과실 상단부(Pedicel이 붙어 있을 것으로 예상되는 영역)의 포인트클라우드에 대해 공분산
    행렬의 최대 고유값에 대응하는 고유벡터를 축 방향으로 사용한다.

    신뢰도는 3D 포인트클라우드 형상 기술자 중 **linearity**를 사용한다:
        linearity = (λ1 - λ2) / (λ1 + λ2 + λ3)      (λ1 ≥ λ2 ≥ λ3)

    "최대 고유값 / 전체 분산" 비율을 쓰지 않는 이유: 그 지표는 막대(rod) 형상과 납작한
    원반(disc) 형상을 구분하지 못한다. Pedicel이 포인트클라우드에 포함되지 않아 과실 상단의
    구면 캡만 남은 경우 형상은 원반이 되는데(λ1 ≈ λ2 >> λ3), 이때 최대 고유값 비율은 약
    0.5로 임계값 근처에 걸려 잘못된 측면 방향 축을 그대로 통과시킨다. 반면 linearity는
    막대일 때 1에 수렴하고 원반일 때 0에 수렴하므로 두 경우를 명확히 분리한다.
    """

    def estimate(self, points_above_fruit_mm: np.ndarray) -> PedicelAxisEstimate:
        _validate_point_cloud(points_above_fruit_mm, name="points_above_fruit_mm")
        if points_above_fruit_mm.shape[0] < 3:
            raise InvalidPointCloudError(
                "Pedicel 축 추정에는 최소 3개 이상의 포인트가 필요합니다."
            )

        origin = points_above_fruit_mm.mean(axis=0)
        centered = points_above_fruit_mm - origin
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # 오름차순 정렬됨

        principal_axis = eigenvectors[:, -1]
        principal_axis = principal_axis / (np.linalg.norm(principal_axis) + 1e-12)

        # 수치 오차로 인한 미세한 음수 고유값을 0으로 클램프(공분산 행렬은 이론상 준정부호).
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        total_variance = float(np.sum(eigenvalues))
        if total_variance > 1e-12:
            linearity = float((eigenvalues[-1] - eigenvalues[-2]) / total_variance)
        else:
            linearity = 0.0

        return PedicelAxisEstimate(
            origin_mm=origin, direction_unit=principal_axis, confidence=linearity
        )
