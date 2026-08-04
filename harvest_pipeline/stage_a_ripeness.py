"""Stage A: 숙성도 판별 게이트.

Stage 0가 만든 과실 인스턴스 마스크를 입력받아 숙성 단계를 판정하고, 수확 대상 여부를
결정한다. 이 단계의 통과 여부가 Stage B(파지·절단 포즈 추정) 실행의 하드 게이트다 —
숙성도 미달 과실에 대해 파지 계획을 세우는 것은 순수한 연산 낭비이므로, Edge 하드웨어
(Jetson Orin NX)의 제한된 자원을 아끼기 위해 여기서 조기 종료시킨다.

중요: "미숙과라서 수확하지 않음"은 오류가 아니라 정상적인 판정 결과다. 따라서 예외를
던지지 않고 GateDecision enum이 담긴 GateResult 객체로 표현한다(exceptions.py 설계 원칙).

참고: 재시도 소진은 GateDecision.SKIP_LOW_CONFIDENCE로 이미 표현되므로, 이를 예외로
승격시키는 별도 헬퍼는 이 모듈에 두지 않는다(과거 존재했으나 호출부가 없어 제거함).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from harvest_pipeline.config import ExceptionReason, RipenessGateConfig, RipenessStage
from harvest_pipeline.exceptions import InvalidImageError
from harvest_pipeline.interfaces import (
    RipenessClassifierModel,
    RipenessPrediction,
    SegmentationInstance,
    sample_multi_patch_regions,
)

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False


class GateDecision(StrEnum):
    """Stage A 판정 결과. HARVEST만 Stage B로 진행한다."""

    HARVEST = "harvest"  # 수확 대상 확정 → Stage B 진행
    SKIP_IMMATURE = "skip_immature"  # 미숙과 → 차기 순회 대상
    SKIP_EXCEPTION = "skip_exception"  # 열과/기형과/과숙 등 예외 → 폐기·가공 트레이
    SKIP_LOW_CONFIDENCE = "skip_low_confidence"  # 재촬영 후에도 신뢰도 미달 → 차기 순회


@dataclass(frozen=True, slots=True)
class GateResult:
    """Stage A의 판정 결과와 그 근거."""

    decision: GateDecision
    fruit_id: str
    ripeness_stage: RipenessStage | None = None
    confidence: float = 0.0
    exception_reason: ExceptionReason | None = None
    recapture_attempts: int = 0
    patch_count: int = 0

    @property
    def should_proceed_to_stage_b(self) -> bool:
        return self.decision is GateDecision.HARVEST

    def as_log_fields(self) -> dict[str, object]:
        """구조화 로깅/KPI 집계용 평탄화 딕셔너리(대시보드가 이 형태를 소비한다)."""
        return {
            "fruit_id": self.fruit_id,
            "decision": self.decision.value,
            "ripeness_stage": self.ripeness_stage.value if self.ripeness_stage else None,
            "confidence": round(self.confidence, 4),
            "exception_reason": self.exception_reason.value if self.exception_reason else None,
            "recapture_attempts": self.recapture_attempts,
            "patch_count": self.patch_count,
        }


def detect_cracking(
    rgb: np.ndarray, mask: np.ndarray, config: RipenessGateConfig
) -> bool:
    """표면 엣지 밀도로 열과(Cracking) 여부를 판정한다(체크리스트 A-7).

    열과는 표피에 균열선이 생기므로 마스크 내부의 Canny 엣지 픽셀 비율이 정상 과실보다
    유의하게 높다. 임계값(cracking_edge_density_threshold)은 품종/조명에 따라 달라지므로
    설정으로 노출한다.

    구현 노트(크기 의존성 버그 수정, 실측으로 확인됨): 마스크 전체에서 엣지를 세면 과실
    윤곽선(테두리) 자체가 배경과 강한 대비를 이루어 Canny 엣지로 잡힌다. 이 테두리 기여는
    둘레/면적 비율(≈2/반지름)로 감소하므로, 균열이 전혀 없어도 작은 과실일수록 엣지 밀도가
    커지는 순수한 크기 의존 아티팩트였다(지름 20px에서 0.10, 100px에서 0.02 — 기존 임계값
    0.12를 지름 17px 미만에서 정상 과실도 초과시킴). 이를 막기 위해:
      1) 마스크를 침식(erode)해 테두리 밴드를 계수 대상에서 제외하고,
      2) 남은 엣지 수를 "원본 마스크 면적"이 아니라 "침식된 마스크 면적"으로 정규화한다.
    테두리 기여가 사라지면 매끄러운 과실 내부에는 (노이즈를 제외하면) 실제 엣지가 거의
    없으므로, 이 지표는 침식된 면적 기준으로 크기와 무관하게 낮게 유지된다 — 반면 실제
    균열선은 침식 후에도 남아 지표를 임계값 이상으로 밀어올린다.
    """
    if not _CV2_AVAILABLE:  # pragma: no cover
        raise ImportError("detect_cracking에는 opencv가 필요합니다.")
    if mask.shape != rgb.shape[:2]:
        raise InvalidImageError(f"mask shape {mask.shape} != rgb shape {rgb.shape[:2]}")

    fruit_pixel_count = int(np.count_nonzero(mask))
    if fruit_pixel_count == 0:
        return False

    # 마스크 경계선이 배경과의 대비로 인해 Canny 엣지로 오검출되는 것을 막기 위해, 침식된
    # 마스크 내부만 검사 대상으로 삼는다. 커널 3~5px 중 5px을 택한 이유: 세그멘테이션
    # 마스크 경계 자체의 1~2px 수준 앤티에일리어싱/노이즈까지 여유 있게 제외하기 위함이다.
    erosion_kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(mask.astype(np.uint8), erosion_kernel, iterations=1).astype(bool)
    eroded_pixel_count = int(np.count_nonzero(eroded_mask))
    if eroded_pixel_count == 0:
        # 과실이 너무 작아 침식 후 내부가 남지 않으면 열과 여부를 판단할 근거 자체가 없다.
        # 이 경우 "열과 아님"으로 처리하는 것이 안전하다 — 판별 불가를 이유로 정상 과실을
        # 오판정(false positive)으로 폐기하는 것보다, 판정을 보류하고 다음 검사 단계
        # (숙성도 분류 등)로 넘기는 편이 낫다.
        return False

    # Canny를 전체 프레임이 아니라 마스크의 bounding box(+여유 패딩)로 crop해 적용한다.
    # 과실 하나는 보통 전체 프레임의 1% 미만을 차지하므로 연산량이 크게 줄어든다. 패딩은
    # crop 경계 자체에서 발생하는 인위적 그라디언트가, 이미 침식으로 제외된 마스크 경계
    # 대역과 겹쳐서 새로운 아티팩트를 만들지 않도록 여유를 두는 것이다.
    ys, xs = np.nonzero(mask)
    pad = 5
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, rgb.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, rgb.shape[1])

    gray_crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    edges_crop = cv2.Canny(gray_crop, 50, 150)
    eroded_mask_crop = eroded_mask[y0:y1, x0:x1]
    edge_pixel_count = int(np.count_nonzero(edges_crop[eroded_mask_crop]))

    # 크기 불변 정규화: 분모를 원본 마스크 면적이 아니라 "침식된 마스크 면적"으로 바꾼다.
    # 테두리 기여를 제거했으므로 매끄러운 과실은 지름 15~100px 전 구간에서 이 지표가 0에
    # 가깝게 유지되며(실측 검증됨), 정규화 계수를 별도로 도입할 필요가 없다 — 기존 임계값
    # 0.12는 이미 "엣지 픽셀 수 / 면적" 스케일로 설계되었고, 이 지표도 동일한 스케일이다.
    edge_density = edge_pixel_count / eroded_pixel_count
    return edge_density > config.cracking_edge_density_threshold


def detect_malformation(mask: np.ndarray, config: RipenessGateConfig) -> bool:
    """원형도(circularity)로 기형과(Catface) 여부를 판정한다(체크리스트 A-8).

    circularity = 4*pi*area / perimeter^2 (완전한 원이면 1.0). 정상 방울토마토는 구형에
    가까워 값이 1에 근접하고, 기형과는 표면이 함몰/주름져 둘레가 길어지므로 값이 낮아진다.
    이 판정이 중요한 이유는, Stage B의 구(Sphere) 피팅이 "과실은 구형"이라는 가정에
    의존하기 때문이다 — 기형과를 미리 걸러야 잘못된 포즈로 파지를 시도하지 않는다.
    """
    if not _CV2_AVAILABLE:  # pragma: no cover
        raise ImportError("detect_malformation에는 opencv가 필요합니다.")

    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True  # 윤곽을 찾을 수 없으면 형상 판단 불가 → 보수적으로 기형과 취급

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, closed=True)
    if perimeter <= 1e-6 or area <= 0:
        return True

    circularity = 4.0 * np.pi * area / (perimeter**2)
    return circularity < config.catface_circularity_min


def evaluate_ripeness_gate(
    rgb: np.ndarray,
    instance: SegmentationInstance,
    classifier: RipenessClassifierModel,
    config: RipenessGateConfig,
    *,
    fruit_id: str,
    recapture_fn: Callable[[], np.ndarray] | None = None,
) -> GateResult:
    """단일 과실에 대해 숙성도 게이트를 평가한다.

    Args:
        rgb: Stage 0를 통과한 보정된 RGB 이미지.
        instance: 해당 과실의 세그멘테이션 인스턴스(마스크 포함).
        classifier: 숙성도 분류 모델(Protocol — CNN 또는 규칙 기반 베이스라인).
        config: 게이트 임계값 설정.
        fruit_id: 로깅/추적용 과실 식별자.
        recapture_fn: 신뢰도 미달 시 재촬영을 수행하는 콜백. None이면 재촬영 없이 즉시
            SKIP_LOW_CONFIDENCE로 판정한다(예: 오프라인 배치 처리 시).

    Returns:
        GateResult: 판정 결과. 예외를 던지지 않으며, 모든 스킵 사유는 decision 필드로 표현된다.
    """
    current_rgb = rgb
    attempts = 0
    last_prediction: RipenessPrediction | None = None
    last_patch_count = 0

    # 예외 과실(열과/기형과) 판정은 색상과 무관한 형태·질감 기반이므로 재촬영 루프 밖에서
    # 한 번만 수행한다(재촬영해도 결과가 바뀌지 않으며, 불필요한 연산 반복을 피한다).
    if detect_cracking(current_rgb, instance.mask, config):
        return GateResult(
            decision=GateDecision.SKIP_EXCEPTION,
            fruit_id=fruit_id,
            exception_reason=ExceptionReason.CRACKING,
        )
    if detect_malformation(instance.mask, config):
        return GateResult(
            decision=GateDecision.SKIP_EXCEPTION,
            fruit_id=fruit_id,
            exception_reason=ExceptionReason.MALFORMED_CATFACE,
        )

    while attempts <= config.max_recapture_retries:
        patches = sample_multi_patch_regions(current_rgb, instance.mask, config)
        if not patches:
            return GateResult(
                decision=GateDecision.SKIP_LOW_CONFIDENCE,
                fruit_id=fruit_id,
                recapture_attempts=attempts,
                patch_count=0,
            )

        last_patch_count = len(patches)
        last_prediction = classifier.predict(patches)

        if last_prediction.confidence >= config.confidence_threshold:
            if last_prediction.stage in config.target_stages:
                decision = GateDecision.HARVEST
            else:
                decision = GateDecision.SKIP_IMMATURE
            return GateResult(
                decision=decision,
                fruit_id=fruit_id,
                ripeness_stage=last_prediction.stage,
                confidence=last_prediction.confidence,
                recapture_attempts=attempts,
                patch_count=last_patch_count,
            )

        attempts += 1
        if attempts > config.max_recapture_retries or recapture_fn is None:
            break
        current_rgb = recapture_fn()

    return GateResult(
        decision=GateDecision.SKIP_LOW_CONFIDENCE,
        fruit_id=fruit_id,
        ripeness_stage=last_prediction.stage if last_prediction else None,
        confidence=last_prediction.confidence if last_prediction else 0.0,
        recapture_attempts=attempts,
        patch_count=last_patch_count,
    )
