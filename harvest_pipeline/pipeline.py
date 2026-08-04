"""전체 전처리 파이프라인 오케스트레이션.

Stage 0(공통) → Stage A(숙성도 게이트) → Stage B(파지·절단 결합 포즈)를 직렬로 연결하고,
과실 단위 예외 격리·타임아웃 감시·구조화 로깅을 담당한다.

핵심 안정성 설계 — 과실 단위 예외 격리:
트러스 하나에는 10~30개의 과실이 있다. 그중 한 개에서 예상치 못한 오류가 발생했다고
트러스 전체 처리를 중단하면, 정상 수확 가능했던 나머지 과실까지 놓치게 되어 미탐지율 KPI가
급격히 악화된다. 따라서 개별 과실 처리는 예외를 잡아 FRUIT_ERROR 상태로 기록하고 다음
과실로 넘어가며, 트러스 전체를 중단시키는 것은 센서 동기화 실패처럼 프레임 자체가 신뢰할 수
없는 경우로 한정한다. 단 프로세스 전체가 위험한 예외(MemoryError 등)는 삼키지 않고
재전파한다 — 그것을 과실 단위 오류로 기록하면 30개 과실이 모두 FRUIT_ERROR가 되면서
"하드웨어 문제였다"는 진짜 원인이 로그에서 은폐된다.

좌표계:
`deproject_depth_to_points`는 **카메라 좌표계**(X 우, Y 하, Z 광축=깊이)를 반환한다.
로봇에 명령을 내리려면 로봇 베이스 좌표계로 변환해야 하며, 그 변환은
`FrameInput.camera_to_base_transform`(4×4 동차행렬)으로 주입한다. 주입하지 않으면 산출된
포즈는 카메라 좌표계 값이므로 그대로 로봇에 보내면 안 된다.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from harvest_pipeline.config import PipelineConfig
from harvest_pipeline.exceptions import (
    HardwareTimeoutError,
    HarvestPipelineError,
    InvalidImageError,
    SegmentationModelError,
)
from harvest_pipeline.interfaces import (
    InstanceSegmentationModel,
    PedicelPoseEstimator,
    RipenessClassifierModel,
    SegmentationInstance,
)
from harvest_pipeline.logging_utils import get_logger, stage_timer
from harvest_pipeline.stage0_common import (
    calibrate_color,
    check_sensor_sync,
    fill_depth_and_denoise,
    mask_static_structures,
    remove_shadow,
    remove_statistical_outliers,
    suppress_specular_highlight,
    voxel_downsample,
)
from harvest_pipeline.stage_a_ripeness import GateResult, evaluate_ripeness_gate
from harvest_pipeline.stage_b_grasp_cut import PoseResult, PoseStatus, compute_grasp_cut_pose

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False

# 프로세스 전체가 손상된 상태를 뜻하는 예외들. 과실 단위 오류로 삼키면 진짜 원인이
# 은폐되므로 격리 경계를 통과시켜 상위로 재전파한다.
_FATAL_EXCEPTIONS = (MemoryError, RecursionError, SystemError)


class FruitOutcome(StrEnum):
    """개별 과실의 최종 처리 결과."""

    READY_TO_HARVEST = "ready_to_harvest"
    SKIPPED_BY_RIPENESS_GATE = "skipped_by_ripeness_gate"
    SKIPPED_BY_POSE_STAGE = "skipped_by_pose_stage"
    FRUIT_ERROR = "fruit_error"


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """핀홀 카메라 내부 파라미터(Depth → 3D 역투영에 사용)."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("초점거리 fx, fy는 양수여야 합니다.")


@dataclass(frozen=True, slots=True)
class FrameInput:
    """파이프라인 한 번 실행에 필요한 입력 묶음."""

    rgb: np.ndarray  # (H, W, 3) uint8
    depth_mm: np.ndarray  # (H, W) float
    intrinsics: CameraIntrinsics
    rgb_timestamp_ms: float
    depth_timestamp_ms: float
    truss_id: str
    static_structure_mask: np.ndarray | None = None  # (H, W) bool
    white_reference_rgb: tuple[float, float, float] | None = None
    """캘리브레이션 시 측정한 ColorChecker 흰색 패치 RGB값. None이면 채널별 화이트밸런스를
    건너뛰고 노출 정규화만 수행한다(stage0_common.calibrate_color 참고)."""
    camera_to_base_transform: np.ndarray | None = None
    """카메라→로봇 베이스 좌표계 변환(4×4 동차행렬). Hand-Eye Calibration 산출물이며,
    주입하면 Stage B의 포즈가 로봇 베이스 좌표계로 산출된다. None이면 카메라 좌표계
    값이므로 로봇 명령으로 직접 사용해서는 안 된다."""

    def validate(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise InvalidImageError(f"rgb는 (H, W, 3) 형태여야 합니다: {self.rgb.shape}")
        if self.rgb.dtype != np.uint8:
            raise InvalidImageError(f"rgb는 uint8이어야 합니다: {self.rgb.dtype}")
        if self.depth_mm.shape != self.rgb.shape[:2]:
            raise InvalidImageError(
                f"depth_mm shape {self.depth_mm.shape}가 rgb의 (H, W) {self.rgb.shape[:2]}와 다릅니다."
            )
        if not np.issubdtype(self.depth_mm.dtype, np.number):
            raise InvalidImageError(f"depth_mm은 수치형이어야 합니다: {self.depth_mm.dtype}")
        if (
            self.static_structure_mask is not None
            and self.static_structure_mask.shape != self.rgb.shape[:2]
        ):
            raise InvalidImageError("static_structure_mask shape이 rgb의 (H, W)와 다릅니다.")
        if self.camera_to_base_transform is not None and self.camera_to_base_transform.shape != (4, 4):
            raise InvalidImageError(
                "camera_to_base_transform은 (4, 4) 동차행렬이어야 합니다: "
                f"{self.camera_to_base_transform.shape}"
            )


@dataclass(frozen=True, slots=True)
class FruitResult:
    """개별 과실의 처리 결과 전체(게이트 결과 + 포즈 결과 + 오류 메시지)."""

    fruit_id: str
    outcome: FruitOutcome
    gate_result: GateResult | None = None
    pose_result: PoseResult | None = None
    error_message: str | None = None

    def as_log_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {"fruit_id": self.fruit_id, "outcome": self.outcome.value}
        if self.gate_result is not None:
            fields["gate"] = self.gate_result.as_log_fields()
        if self.pose_result is not None:
            fields["pose"] = self.pose_result.as_log_fields()
        if self.error_message is not None:
            fields["error_message"] = self.error_message
        return fields


@dataclass(frozen=True, slots=True)
class TrussResult:
    """트러스 한 개 처리 결과 및 KPI 집계."""

    truss_id: str
    fruit_results: tuple[FruitResult, ...] = field(default_factory=tuple)
    detected_fruit_count: int = 0
    elapsed_ms: float = 0.0
    aborted_reason: str | None = None
    timed_out: bool = False

    @property
    def harvestable(self) -> tuple[FruitResult, ...]:
        return tuple(r for r in self.fruit_results if r.outcome is FruitOutcome.READY_TO_HARVEST)

    def kpi_summary(self) -> dict[str, object]:
        """PRD 3장 KPI 대시보드가 소비하는 집계 지표.

        미탐지율(1회 순회)은 "탐지되었으나 수확 불가로 스킵된 비율"로 근사한다. 카메라에
        아예 보이지 않아 세그멘테이션조차 되지 않은 과실은 정의상 이 프레임에서 셀 수
        없으므로, 실제 KPI 산출 시에는 트러스별 실제 과실 총수(작업자 실측 또는 누적
        재순회 기록)를 분모로 사용해 보정해야 한다.
        """
        total = len(self.fruit_results)
        ready = len(self.harvestable)
        return {
            "truss_id": self.truss_id,
            "detected_fruit_count": self.detected_fruit_count,
            "evaluated_fruit_count": total,
            "ready_to_harvest_count": ready,
            "skipped_by_ripeness_gate": sum(
                1 for r in self.fruit_results if r.outcome is FruitOutcome.SKIPPED_BY_RIPENESS_GATE
            ),
            "skipped_by_pose_stage": sum(
                1 for r in self.fruit_results if r.outcome is FruitOutcome.SKIPPED_BY_POSE_STAGE
            ),
            "fruit_error_count": sum(
                1 for r in self.fruit_results if r.outcome is FruitOutcome.FRUIT_ERROR
            ),
            "ready_ratio": round(ready / total, 4) if total else 0.0,
            "unprocessed_count": max(self.detected_fruit_count - total, 0),
            "timed_out": self.timed_out,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "aborted_reason": self.aborted_reason,
        }


def deproject_depth_to_points(
    depth_mm: np.ndarray, mask: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """마스크 영역의 Depth 픽셀을 카메라 좌표계 3D 포인트로 역투영한다.

    핀홀 모델: X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy,  Z = depth
    마스크된 픽셀만 인덱싱해 처리하므로 전체 프레임을 역투영하는 것보다 메모리·연산이
    훨씬 적다(과실 하나는 보통 전체 프레임의 1% 미만).
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth_mm[ys, xs].astype(np.float32)
    valid = z > 0
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    ys, xs, z = ys[valid], xs[valid], z[valid]
    x = (xs.astype(np.float32) - intrinsics.cx) * z / intrinsics.fx
    y = (ys.astype(np.float32) - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=-1)


def transform_points(points_mm: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """4×4 동차행렬로 포인트클라우드를 좌표 변환한다(카메라 → 로봇 베이스)."""
    if points_mm.size == 0:
        return points_mm
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (points_mm @ rotation.T + translation).astype(np.float32)


def estimate_visible_ratio(instance: SegmentationInstance) -> float:
    """최소외접원 면적 대비 마스크 면적으로 가시 비율을 추정한다.

    구형 과실이 온전히 보이면 마스크는 최소외접원을 거의 채우므로 비율이 1에 가깝다. 잎에
    가려져 일부가 잘리면 최소외접원의 지름은 유지되는 반면 마스크 면적만 줄어들어 비율이
    떨어진다(정확히 절반이 가려지면 약 0.5).

    이전 구현(마스크 면적 / bbox 면적)을 쓰지 않는 이유: bbox가 그 마스크로부터 도출되기
    때문에 가림으로 마스크가 줄면 bbox도 함께 줄어 비율이 보존된다. 실측 결과 우측 절반이
    가려진 과실이 0.993, 온전한 과실이 0.965로 **가려진 쪽이 더 높게** 나와 occlusion
    게이트가 전혀 발동하지 않았다.
    """
    if not _CV2_AVAILABLE:  # pragma: no cover
        raise ImportError("estimate_visible_ratio에는 opencv가 필요합니다.")

    mask_area = int(np.count_nonzero(instance.mask))
    if mask_area == 0:
        return 0.0

    contours, _ = cv2.findContours(
        instance.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    _, radius = cv2.minEnclosingCircle(largest)
    enclosing_area = float(np.pi * radius**2)
    if enclosing_area <= 1e-6:
        return 0.0

    return float(min(mask_area / enclosing_area, 1.0))


class HarvestPreprocessingPipeline:
    """방울토마토 수확 전처리 파이프라인.

    모델 컴포넌트(세그멘테이션/숙성도 분류/Pedicel 축 추정)를 생성자에서 주입받으므로,
    규칙 기반 베이스라인과 학습된 모델을 코드 수정 없이 교체할 수 있다(의존성 역전).

    리소스 관리: 비전 추론 타임아웃을 위해 단일 워커 스레드 풀을 유지한다. 컨텍스트
    매니저로 사용하거나 `close()`를 호출해 정리할 수 있다.
    """

    def __init__(
        self,
        config: PipelineConfig,
        segmentation_model: InstanceSegmentationModel,
        ripeness_classifier: RipenessClassifierModel,
        pedicel_estimator: PedicelPoseEstimator,
        *,
        logger_name: str = "harvest_pipeline",
    ) -> None:
        self._config = config
        self._segmentation_model = segmentation_model
        self._ripeness_classifier = ripeness_classifier
        self._pedicel_estimator = pedicel_estimator
        self._logger = get_logger(logger_name, config.logging)
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="harvest-vision"
        )

    def close(self) -> None:
        """워커 스레드 풀을 정리한다. 진행 중인 추론이 끝날 때까지 기다리지 않는다."""
        self._inference_executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "HarvestPreprocessingPipeline":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Stage 0
    # ------------------------------------------------------------------
    def run_stage0(self, frame: FrameInput) -> tuple[np.ndarray, np.ndarray]:
        """공통 전처리를 수행해 보정된 (RGB, Depth)를 반환한다.

        Raises:
            SensorSyncError: 센서 동기화 실패(프레임 전체를 신뢰할 수 없어 상위에서 중단).
            InvalidImageError: 입력 형식 불일치.
        """
        frame.validate()
        cfg = self._config

        with stage_timer(self._logger, "stage0_common", truss_id=frame.truss_id) as ctx:
            check_sensor_sync(frame.rgb_timestamp_ms, frame.depth_timestamp_ms, cfg.sensor_sync)

            rgb = calibrate_color(
                frame.rgb,
                cfg.color_calibration,
                white_reference_rgb=frame.white_reference_rgb,
            )
            rgb = suppress_specular_highlight(rgb, cfg.highlight_suppression)
            rgb = remove_shadow(rgb, cfg.shadow_removal)
            depth = fill_depth_and_denoise(frame.depth_mm, cfg.depth_filter)

            if frame.static_structure_mask is not None:
                rgb = mask_static_structures(rgb, frame.static_structure_mask)
                depth = mask_static_structures(depth, frame.static_structure_mask)
                ctx["static_structures_masked"] = True

            ctx["rgb_shape"] = list(rgb.shape)
        return rgb, depth

    def segment(self, rgb: np.ndarray, *, truss_id: str) -> tuple[SegmentationInstance, ...]:
        """인스턴스 세그멘테이션을 수행한다(Stage 0 말단 — A/B 양쪽이 공유).

        타임아웃 한계(중요): 추론을 워커 스레드에서 실행하고 `future.result(timeout=...)`로
        **호출자를 해제**하므로 파이프라인은 제한 시간 내에 제어권을 되찾는다. 그러나 파이썬
        스레드는 강제 종료할 수 없어 **워커 스레드 자체는 계속 실행된다**(협조적 취소 불가).
        따라서 연속 타임아웃이 반복되면 스레드가 누적되므로, 운영 환경에서는 상위 감시자가
        일정 횟수 이후 프로세스를 재시작해야 한다. 진짜 강제 취소가 필요하면 추론을 별도
        프로세스로 분리해 terminate()해야 하지만, RGB-D 프레임 직렬화 비용과 모델 재로드
        비용이 크므로 현재는 스레드 방식을 택했다.

        Raises:
            SegmentationModelError: 모델 추론이 실패한 경우.
            HardwareTimeoutError: 제한 시간 내에 결과를 얻지 못한 경우.
        """
        timeout_s = self._config.timeouts.vision_inference_timeout_s

        with stage_timer(self._logger, "segmentation", truss_id=truss_id) as ctx:
            future = self._inference_executor.submit(self._segmentation_model.predict, rgb)
            try:
                result = future.result(timeout=timeout_s)
            except FutureTimeoutError as exc:
                future.cancel()
                raise HardwareTimeoutError(
                    f"비전 추론이 타임아웃({timeout_s}s)을 초과했습니다. "
                    "워커 스레드는 강제 종료할 수 없어 백그라운드에서 계속 실행됩니다."
                ) from exc
            except HarvestPipelineError:
                raise
            except _FATAL_EXCEPTIONS:
                raise
            except Exception as exc:  # 외부 모델 라이브러리의 임의 예외를 도메인 예외로 변환
                raise SegmentationModelError(f"세그멘테이션 추론 실패: {exc}") from exc

            fruits = result.fruit_instances
            ctx["detected_fruit_count"] = len(fruits)
            ctx["detected_instance_count"] = len(result.instances)
        return fruits

    # ------------------------------------------------------------------
    # 단일 과실 처리 (Stage A → Stage B)
    # ------------------------------------------------------------------
    def process_fruit(
        self,
        rgb: np.ndarray,
        depth_mm: np.ndarray,
        instance: SegmentationInstance,
        intrinsics: CameraIntrinsics,
        *,
        truss_id: str,
        fruit_id: str,
        obstacle_points_mm: np.ndarray | None = None,
        recapture_fn: Callable[[], np.ndarray] | None = None,
        camera_to_base_transform: np.ndarray | None = None,
    ) -> FruitResult:
        """한 개 과실에 대해 Stage A 게이트와 Stage B 포즈 추정을 순차 수행한다.

        예외를 밖으로 전파하지 않고 FruitResult(FRUIT_ERROR)로 변환한다 — 과실 하나의
        오류로 트러스 전체 순회가 중단되는 것을 막기 위한 격리 경계다. 단 프로세스 전체가
        위험한 예외(_FATAL_EXCEPTIONS)는 재전파한다.
        """
        try:
            with stage_timer(
                self._logger, "stage_a_ripeness", truss_id=truss_id, fruit_id=fruit_id
            ) as ctx:
                gate_result = evaluate_ripeness_gate(
                    rgb,
                    instance,
                    self._ripeness_classifier,
                    self._config.ripeness_gate,
                    fruit_id=fruit_id,
                    recapture_fn=recapture_fn,
                )
                ctx.update(gate_result.as_log_fields())

            if not gate_result.should_proceed_to_stage_b:
                return FruitResult(
                    fruit_id=fruit_id,
                    outcome=FruitOutcome.SKIPPED_BY_RIPENESS_GATE,
                    gate_result=gate_result,
                )

            with stage_timer(
                self._logger, "stage_b_grasp_cut", truss_id=truss_id, fruit_id=fruit_id
            ) as ctx:
                # 과실 본체와 Pedicel 포인트를 분리해 전달한다 — 구 피팅에는 본체만,
                # 축 추정에는 Pedicel만 사용해야 한다(stage_b_grasp_cut 참고).
                body_points = self._extract_points(
                    depth_mm, instance.mask, intrinsics, camera_to_base_transform
                )
                pedicel_points = (
                    self._extract_points(
                        depth_mm, instance.pedicel_mask, intrinsics, camera_to_base_transform
                    )
                    if instance.pedicel_mask is not None
                    else None
                )

                visible_ratio = estimate_visible_ratio(instance)
                pose_result = compute_grasp_cut_pose(
                    body_points,
                    self._config.grasp_cut_pose,
                    self._pedicel_estimator,
                    fruit_id=fruit_id,
                    visible_ratio=visible_ratio,
                    pedicel_points_mm=pedicel_points,
                    obstacle_points_mm=obstacle_points_mm,
                )
                ctx.update(pose_result.as_log_fields())
                ctx["body_point_count"] = int(body_points.shape[0])
                ctx["pedicel_point_count"] = (
                    int(pedicel_points.shape[0]) if pedicel_points is not None else 0
                )

            outcome = (
                FruitOutcome.READY_TO_HARVEST
                if pose_result.status is PoseStatus.READY
                else FruitOutcome.SKIPPED_BY_POSE_STAGE
            )
            return FruitResult(
                fruit_id=fruit_id,
                outcome=outcome,
                gate_result=gate_result,
                pose_result=pose_result,
            )

        except _FATAL_EXCEPTIONS:
            # 프로세스 전체가 손상된 상태다. 과실 단위 오류로 기록하면 30개 과실이 모두
            # FRUIT_ERROR가 되면서 진짜 원인(메모리 고갈 등)이 로그에서 은폐된다.
            self._logger.exception(
                "fatal_error_during_fruit_processing",
                extra={"truss_id": truss_id, "fruit_id": fruit_id},
            )
            raise
        except Exception as exc:
            # 개별 과실의 실패는 트러스 순회를 중단시키지 않는다(위 클래스 docstring 참고).
            self._logger.exception(
                "fruit_processing_failed",
                extra={"truss_id": truss_id, "fruit_id": fruit_id, "error_type": type(exc).__name__},
            )
            return FruitResult(
                fruit_id=fruit_id,
                outcome=FruitOutcome.FRUIT_ERROR,
                error_message=f"{type(exc).__name__}: {exc}",
            )

    def _extract_points(
        self,
        depth_mm: np.ndarray,
        mask: np.ndarray,
        intrinsics: CameraIntrinsics,
        camera_to_base_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        """마스크 영역을 역투영하고 다운샘플링·이상치 제거까지 수행한다."""
        points = deproject_depth_to_points(depth_mm, mask, intrinsics)
        points = voxel_downsample(points, self._config.point_cloud)
        points = remove_statistical_outliers(points, self._config.point_cloud)
        if camera_to_base_transform is not None:
            points = transform_points(points, camera_to_base_transform)
        return points

    # ------------------------------------------------------------------
    # 트러스 단위 실행
    # ------------------------------------------------------------------
    def process_truss(
        self,
        frame: FrameInput,
        *,
        recapture_fn: Callable[[], np.ndarray] | None = None,
    ) -> TrussResult:
        """트러스 한 개(프레임 하나)를 처리해 수확 대상 목록과 KPI를 반환한다.

        프레임 전체를 신뢰할 수 없는 오류(센서 동기화 실패, 입력 형식 오류, 세그멘테이션
        모델 장애, OpenCV 내부 오류 등)는 aborted_reason에 기록하고 빈 결과를 반환한다.
        개별 과실 오류는 process_fruit 내부에서 격리되어 여기까지 전파되지 않는다.
        """
        started = time.perf_counter()

        try:
            rgb, depth = self.run_stage0(frame)
            fruit_instances = self.segment(rgb, truss_id=frame.truss_id)
        except _FATAL_EXCEPTIONS:
            raise
        except Exception as exc:
            # HarvestPipelineError뿐 아니라 cv2.error, ImportError, numpy 예외까지 잡는다.
            # 좁게 잡으면 이런 예외가 그대로 뚫고 나가 "프레임 오류는 aborted_reason에
            # 기록한다"는 이 함수의 계약이 깨진다.
            self._logger.error(
                "truss_aborted",
                extra={
                    "truss_id": frame.truss_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return TrussResult(
                truss_id=frame.truss_id,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                aborted_reason=f"{type(exc).__name__}: {exc}",
            )

        # 인접 과실과 잎/줄기를 모두 장애물로 사용한다. 과실만 넘기면 실제로 접근 경로를
        # 막는 잎·줄기·유인끈이 충돌 검사에서 완전히 빠진다.
        obstacle_pool, obstacle_owner_ids = self._build_obstacle_pool(
            depth, fruit_instances, frame.intrinsics, frame.camera_to_base_transform
        )

        results: list[FruitResult] = []
        timed_out = False
        for index, instance in enumerate(fruit_instances):
            elapsed_s = time.perf_counter() - started
            if elapsed_s > self._config.timeouts.truss_processing_timeout_s:
                timed_out = True
                self._logger.warning(
                    "truss_timeout_partial_result",
                    extra={
                        "truss_id": frame.truss_id,
                        "processed_count": len(results),
                        "remaining_count": len(fruit_instances) - index,
                        "elapsed_s": round(elapsed_s, 2),
                    },
                )
                break

            fruit_id = f"{frame.truss_id}-F{index:03d}"
            obstacles = self._obstacles_excluding_target(obstacle_pool, obstacle_owner_ids, index)
            results.append(
                self.process_fruit(
                    rgb,
                    depth,
                    instance,
                    frame.intrinsics,
                    truss_id=frame.truss_id,
                    fruit_id=fruit_id,
                    obstacle_points_mm=obstacles,
                    recapture_fn=recapture_fn,
                    camera_to_base_transform=frame.camera_to_base_transform,
                )
            )

        truss_result = TrussResult(
            truss_id=frame.truss_id,
            fruit_results=tuple(results),
            detected_fruit_count=len(fruit_instances),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            timed_out=timed_out,
        )
        self._logger.info("truss_completed", extra=truss_result.kpi_summary())
        return truss_result

    def _build_obstacle_pool(
        self,
        depth_mm: np.ndarray,
        fruit_instances: tuple[SegmentationInstance, ...],
        intrinsics: CameraIntrinsics,
        camera_to_base_transform: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """모든 장애물 포인트를 **단일 배열 한 쌍**으로 미리 구축한다.

        `(points, owner_ids)` 형태로 반환하며 owner_ids는 각 점을 소유한 과실 인덱스다
        (과실이 아닌 잎/줄기 등은 -1). 과실별로 "자기 자신을 제외한 나머지"를 만들 때
        `owner_ids != index` 불리언 마스크만 적용하면 되므로, 과실마다 np.vstack을 호출해
        O(N²) 크기의 메모리를 복사하던 방식을 피할 수 있다(N=30이면 vstack 30회 ×
        각 O(N) 복사 → 단일 배열 1회 구축).
        """
        point_chunks: list[np.ndarray] = []
        owner_chunks: list[np.ndarray] = []

        for index, instance in enumerate(fruit_instances):
            points = self._extract_points(
                depth_mm, instance.combined_mask, intrinsics, camera_to_base_transform
            )
            if points.shape[0] == 0:
                continue
            point_chunks.append(points)
            owner_chunks.append(np.full(points.shape[0], index, dtype=np.int32))

        if not point_chunks:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

        return np.vstack(point_chunks), np.concatenate(owner_chunks)

    @staticmethod
    def _obstacles_excluding_target(
        obstacle_pool: np.ndarray, owner_ids: np.ndarray, target_index: int
    ) -> np.ndarray | None:
        """대상 과실이 소유한 점을 제외한 장애물 포인트를 반환한다(뷰 기반 불리언 인덱싱)."""
        if obstacle_pool.shape[0] == 0:
            return None
        keep = owner_ids != target_index
        if not np.any(keep):
            return None
        return obstacle_pool[keep]
