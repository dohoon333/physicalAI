"""Stage B: 파지·절단 결합 포즈(Grasp-Cut Pose) 추정.

핵심 설계 원칙 — 파지점과 절단점을 독립적으로 추정하지 않는다.
소프트 그리퍼와 가위형 절단날은 하나의 강체로 결합된 복합 엔드이펙터이므로, 둘 사이의
상대 오프셋은 로봇이 매 프레임 추정할 변수가 아니라 하드웨어 제작 시 확정된 상수다
(GraspCutPoseConfig.blade_axial_offset_mm). 따라서 실제로 추정하는 값은 두 가지뿐이다:
  1. 과실 중심 3D 위치 (구 피팅)
  2. Pedicel 축 방향 (PCA 또는 학습된 키포인트 회귀)

기구학 모델 — 측면 접근(side approach):
  로봇은 Pedicel 축에 **수직인** 방향(카메라 쪽)에서 진입한다. 축과 평행하게 진입하면
  그리퍼가 과실을 관통해야 칼날이 Pedicel에 닿으므로 물리적으로 불가능하다.
  - TCP(그리퍼 중심) 목표 = 과실 중심
  - 절단날 위치 = TCP + Pedicel축 × blade_axial_offset_mm  (하드웨어 상수로 유도)
  - pre-grasp 지점 = TCP − 접근방향 × approach_clearance_mm  (충돌 검사 구간)
  회전은 "접근 방향(tool Z) + Pedicel 축(tool Y)" 두 축으로 완전히 결정되므로, 산출물은
  위치 3 + 회전 3 = 진짜 6-DOF다.

좌표계 주의:
  본 모듈의 입력 포인트는 **카메라 좌표계**(X 우, Y 하, Z 광축 = 깊이)를 전제한다.
  카메라에서 멀어지는 방향이 +Z이며, 이미지상 "위"는 −Y다. Depth 센서는 과실의 카메라
  대향면만 관측하므로 과실 뒤쪽(+Z 반구) 포인트는 원리적으로 존재하지 않는다 — 이 때문에
  Pedicel 축은 반드시 별도의 Pedicel 마스크 포인트로 추정해야 하며, 과실 본체 포인트에서
  "위쪽 절반"을 뽑아 대체할 수 없다(그 시도는 항상 0개의 점을 반환한다).
  로봇 베이스 좌표계로의 변환은 파이프라인 상위 계층(FrameInput.camera_to_base_transform)이
  담당한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from harvest_pipeline.config import GraspCutPoseConfig
from harvest_pipeline.exceptions import InvalidPointCloudError
from harvest_pipeline.interfaces import PedicelPoseEstimator


class PoseStatus(StrEnum):
    """Stage B 산출 결과 상태. READY만 로봇 실행으로 넘어간다."""

    READY = "ready"
    SKIP_INSUFFICIENT_POINTS = "skip_insufficient_points"
    SKIP_OCCLUDED = "skip_occluded"
    SKIP_SIZE_OUT_OF_RANGE = "skip_size_out_of_range"
    SKIP_COLLISION = "skip_collision"
    SKIP_LOW_AXIS_CONFIDENCE = "skip_low_axis_confidence"
    SKIP_NO_PEDICEL = "skip_no_pedicel"
    SKIP_UNREACHABLE_OFFSET = "skip_unreachable_offset"


@dataclass(frozen=True, slots=True)
class SphereFitResult:
    center_mm: np.ndarray  # shape (3,)
    radius_mm: float
    rmse_mm: float


@dataclass(frozen=True, slots=True)
class GraspCutPose:
    """로봇에게 전달되는 최종 결과물 — 단일 6-DOF 접근 포즈.

    회전은 `rotation_matrix`(열벡터가 tool X/Y/Z 축)로 명시한다. 위치만으로는 가위 절단면의
    roll이 결정되지 않아 실제 로봇 명령을 만들 수 없기 때문이다.
    """

    grasp_position_mm: np.ndarray  # 그리퍼 TCP 최종 목표(= 과실 중심) (3,)
    pre_grasp_position_mm: np.ndarray  # 접근 시작 지점 (3,)
    approach_direction_unit: np.ndarray  # 로봇 진행 방향(pre_grasp → grasp) (3,)
    cut_position_mm: np.ndarray  # 하드웨어 오프셋으로 유도된 절단날 위치 (3,)
    pedicel_axis_unit: np.ndarray  # 과실 중심에서 Pedicel 쪽으로 향하는 단위 벡터 (3,)
    rotation_matrix: np.ndarray  # (3, 3) — tool 좌표계 기저(열: X, Y, Z)
    fruit_center_mm: np.ndarray
    fruit_radius_mm: float

    def as_log_fields(self) -> dict[str, object]:
        return {
            "grasp_position_mm": [round(float(v), 2) for v in self.grasp_position_mm],
            "pre_grasp_position_mm": [round(float(v), 2) for v in self.pre_grasp_position_mm],
            "approach_direction_unit": [round(float(v), 4) for v in self.approach_direction_unit],
            "cut_position_mm": [round(float(v), 2) for v in self.cut_position_mm],
            "pedicel_axis_unit": [round(float(v), 4) for v in self.pedicel_axis_unit],
            "fruit_radius_mm": round(self.fruit_radius_mm, 2),
        }


@dataclass(frozen=True, slots=True)
class PoseResult:
    """Stage B 결과. 스킵은 예외가 아닌 status 필드로 표현한다."""

    status: PoseStatus
    fruit_id: str
    pose: GraspCutPose | None = None
    visible_ratio: float = 0.0
    axis_confidence: float = 0.0
    sphere_rmse_mm: float = 0.0
    cut_offset_error_mm: float = 0.0

    @property
    def is_executable(self) -> bool:
        return self.status is PoseStatus.READY and self.pose is not None

    def as_log_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "fruit_id": self.fruit_id,
            "status": self.status.value,
            "visible_ratio": round(self.visible_ratio, 4),
            "axis_confidence": round(self.axis_confidence, 4),
            "sphere_rmse_mm": round(self.sphere_rmse_mm, 3),
            "cut_offset_error_mm": round(self.cut_offset_error_mm, 3),
        }
        if self.pose is not None:
            fields.update(self.pose.as_log_fields())
        return fields


def fit_sphere_least_squares(points_mm: np.ndarray) -> SphereFitResult:
    """선형 최소제곱법으로 구의 중심과 반지름을 추정한다.

    구의 방정식 (x-a)² + (y-b)² + (z-c)² = r² 을 전개하면
        2ax + 2by + 2cz + (r² - a² - b² - c²) = x² + y² + z²
    가 되어 미지수 (a, b, c, d)에 대한 선형 시스템 A·w = f 로 변환된다.
    반복 최적화 없이 한 번의 lstsq 호출로 해를 얻으므로 Edge 하드웨어에서도 매우 빠르다
    (RANSAC 대비 이상치에는 약하지만, 앞선 Stage 0의 Statistical Outlier Removal이
    이미 이상치를 제거했다는 전제 하에 성립한다).

    Raises:
        InvalidPointCloudError: 포인트 형식이 잘못되었거나 수치적으로 해를 구할 수 없을 때.
    """
    if points_mm.ndim != 2 or points_mm.shape[1] != 3:
        raise InvalidPointCloudError(
            f"points_mm은 (N, 3) 형태여야 합니다. 실제 shape={points_mm.shape}"
        )
    if points_mm.shape[0] < 4:
        raise InvalidPointCloudError("구 피팅에는 최소 4개의 점이 필요합니다.")

    design = np.hstack([2.0 * points_mm, np.ones((points_mm.shape[0], 1))])
    target = np.sum(points_mm**2, axis=1)

    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    center = solution[:3]
    radius_squared = solution[3] + float(np.sum(center**2))
    if radius_squared <= 0:
        raise InvalidPointCloudError(
            "구 피팅 결과 반지름이 음수/0입니다(포인트 분포가 구형이 아닐 가능성)."
        )
    radius = float(np.sqrt(radius_squared))

    residuals = np.linalg.norm(points_mm - center, axis=1) - radius
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return SphereFitResult(center_mm=center, radius_mm=radius, rmse_mm=rmse)


def _orient_axis_outward(
    axis_direction: np.ndarray, axis_origin: np.ndarray, fruit_center: np.ndarray
) -> np.ndarray:
    """PCA 고유벡터의 임의 부호를 "과실 중심에서 바깥쪽"으로 정규화한다.

    이 보정 없이는 절단 위치가 과실 내부로 계산될 수 있다.
    """
    if float(np.dot(axis_direction, axis_origin - fruit_center)) < 0:
        return -axis_direction
    return axis_direction


def _build_tool_rotation(
    approach_direction: np.ndarray, pedicel_axis: np.ndarray
) -> np.ndarray:
    """접근 방향과 Pedicel 축으로 tool 좌표계 회전행렬을 구성한다(Gram-Schmidt).

    - tool Z = 접근 방향(로봇 진행 방향)
    - tool Y = Pedicel 축에서 Z 성분을 제거해 직교화한 방향(가위 절단면이 Pedicel을 가로지름)
    - tool X = Y × Z (오른손 좌표계)

    두 축이 거의 평행하면 직교화가 수치적으로 불안정하므로, 그 경우 임의의 보조 벡터로
    대체한다(호출부가 이미 수직 접근 방향을 만들어 넘기므로 실제로는 발생하지 않는다).
    """
    tool_z = approach_direction / (np.linalg.norm(approach_direction) + 1e-12)

    tool_y = pedicel_axis - np.dot(pedicel_axis, tool_z) * tool_z
    norm_y = float(np.linalg.norm(tool_y))
    if norm_y < 1e-6:
        # 접근 방향과 Pedicel 축이 평행 — 임의의 직교 벡터를 선택한다.
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(fallback, tool_z))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        tool_y = fallback - np.dot(fallback, tool_z) * tool_z
        norm_y = float(np.linalg.norm(tool_y))
    tool_y = tool_y / norm_y

    tool_x = np.cross(tool_y, tool_z)
    tool_x = tool_x / (np.linalg.norm(tool_x) + 1e-12)

    return np.column_stack([tool_x, tool_y, tool_z])


def _compute_side_approach_direction(pedicel_axis: np.ndarray) -> np.ndarray:
    """Pedicel 축에 수직이면서 카메라에서 과실로 향하는(+Z) 접근 방향을 만든다.

    카메라 좌표계에서 로봇은 카메라와 같은 쪽에서 진입하므로 기본 진행 방향은 +Z(깊이 증가)다.
    거기서 Pedicel 축 성분을 제거해 축에 수직인 성분만 남긴다 — 축과 평행하게 진입하면
    그리퍼가 과실을 관통해야 칼날이 Pedicel에 닿게 되어 물리적으로 불가능하기 때문이다.
    """
    camera_forward = np.array([0.0, 0.0, 1.0])
    perpendicular = camera_forward - np.dot(camera_forward, pedicel_axis) * pedicel_axis
    norm = float(np.linalg.norm(perpendicular))
    if norm < 1e-6:
        # Pedicel 축이 광축과 거의 평행(과실이 카메라를 정면으로 향한 드문 경우).
        # 이때는 이미지 평면 내 임의 방향으로 접근한다.
        perpendicular = np.array([1.0, 0.0, 0.0])
        perpendicular = perpendicular - np.dot(perpendicular, pedicel_axis) * pedicel_axis
        norm = float(np.linalg.norm(perpendicular))
    return perpendicular / norm


def compute_grasp_cut_pose(
    fruit_body_points_mm: np.ndarray,
    config: GraspCutPoseConfig,
    pedicel_estimator: PedicelPoseEstimator,
    *,
    fruit_id: str,
    visible_ratio: float,
    pedicel_points_mm: np.ndarray | None = None,
    obstacle_points_mm: np.ndarray | None = None,
) -> PoseResult:
    """단일 과실에 대한 파지·절단 결합 6-DOF 포즈를 산출한다.

    Args:
        fruit_body_points_mm: **과실 본체만**의 포인트클라우드(카메라 좌표계, mm).
            Pedicel 포인트를 여기에 섞으면 안 된다 — Pedicel은 구면 위에 놓이지 않으므로
            구 피팅의 반지름을 과대 추정하게 만들고, 실제 크기가 정상인 과실이 허용 범위
            초과로 잘못 걸러진다(12mm 과실이 17mm로 추정되는 문제가 확인되었다).
        config: 파지·절단 설정(허용 지름, 절단창, 안전 마진 등).
        pedicel_estimator: Pedicel 축 추정기(Protocol — PCA 또는 학습 모델).
        fruit_id: 로깅/추적용 식별자.
        visible_ratio: 가시 비율(0~1). 가림 판정에 사용.
        pedicel_points_mm: Pedicel 영역만의 포인트클라우드. **필수**다. Depth 센서는 과실의
            카메라 대향면만 관측하므로 과실 본체 포인트로부터 Pedicel 방향을 유추할 수 없다
            (구면 캡은 납작한 원반이라 주축이 측면으로 잡힌다). 없으면 SKIP_NO_PEDICEL.
        obstacle_points_mm: 인접 과실·잎·줄기·구조물 포인트. None이면 충돌 검사를 생략한다.

    Returns:
        PoseResult: 실행 가능 여부(status)와 산출된 포즈. 예외를 던지지 않는다.
    """
    if visible_ratio < config.occlusion_visible_ratio_min:
        return PoseResult(
            status=PoseStatus.SKIP_OCCLUDED, fruit_id=fruit_id, visible_ratio=visible_ratio
        )

    if fruit_body_points_mm.shape[0] < config.min_points_for_sphere_fit:
        return PoseResult(
            status=PoseStatus.SKIP_INSUFFICIENT_POINTS,
            fruit_id=fruit_id,
            visible_ratio=visible_ratio,
        )

    if pedicel_points_mm is None or pedicel_points_mm.shape[0] < 3:
        # 과실 본체에서 "위쪽"을 뽑아 대체하지 않는다 — 모듈 docstring의 좌표계 주의 참고.
        return PoseResult(
            status=PoseStatus.SKIP_NO_PEDICEL,
            fruit_id=fruit_id,
            visible_ratio=visible_ratio,
        )

    try:
        sphere = fit_sphere_least_squares(fruit_body_points_mm)
    except InvalidPointCloudError:
        # 구형 가정이 성립하지 않는 포인트 분포(기형과 등). Stage A에서 대부분 걸러지지만
        # 3D 형상 기준으로도 다시 확인되는 경우가 있어 여기서 스킵 처리한다.
        return PoseResult(
            status=PoseStatus.SKIP_SIZE_OUT_OF_RANGE,
            fruit_id=fruit_id,
            visible_ratio=visible_ratio,
        )

    diameter_mm = sphere.radius_mm * 2.0
    min_diameter, max_diameter = config.fruit_diameter_range_mm
    if not (min_diameter <= diameter_mm <= max_diameter):
        return PoseResult(
            status=PoseStatus.SKIP_SIZE_OUT_OF_RANGE,
            fruit_id=fruit_id,
            visible_ratio=visible_ratio,
            sphere_rmse_mm=sphere.rmse_mm,
        )

    axis = pedicel_estimator.estimate(pedicel_points_mm)

    if axis.confidence < config.min_axis_confidence:
        # 축 방향이 불확실한 상태로 절단을 시도하면 과실 어깨나 모주(줄기)를 손상시킬 수 있어
        # (PRD 6장 "Pedicel 절단 실패로 인한 모주 손상" 리스크) 즉시 스킵한다.
        return PoseResult(
            status=PoseStatus.SKIP_LOW_AXIS_CONFIDENCE,
            fruit_id=fruit_id,
            visible_ratio=visible_ratio,
            axis_confidence=axis.confidence,
            sphere_rmse_mm=sphere.rmse_mm,
        )

    pedicel_axis = _orient_axis_outward(axis.direction_unit, axis.origin_mm, sphere.center_mm)

    # 이상적 절단 거리: 과실 표면에서 절단창(5~10mm) 중앙까지.
    cut_window_min, cut_window_max = config.pedicel_cut_window_mm
    ideal_cut_distance = sphere.radius_mm + (cut_window_min + cut_window_max) / 2.0

    # 하드웨어 오프셋은 고정 상수이므로 이상적 거리와 일치하지 않을 수 있다. 불일치가 허용
    # 범위를 넘으면 칼날이 절단창을 벗어나므로(과실 어깨 또는 모주 손상 위험) 스킵한다.
    offset_error = abs(config.blade_axial_offset_mm - ideal_cut_distance)
    if offset_error > config.blade_offset_tolerance_mm:
        return PoseResult(
            status=PoseStatus.SKIP_UNREACHABLE_OFFSET,
            fruit_id=fruit_id,
            visible_ratio=visible_ratio,
            axis_confidence=axis.confidence,
            sphere_rmse_mm=sphere.rmse_mm,
            cut_offset_error_mm=offset_error,
        )

    # TCP는 과실 중심을 목표로 하고, 절단날 위치는 하드웨어 오프셋으로 유도된다.
    grasp_position = sphere.center_mm
    cut_position = grasp_position + pedicel_axis * config.blade_axial_offset_mm

    approach_direction = _compute_side_approach_direction(pedicel_axis)
    pre_grasp_position = grasp_position - approach_direction * config.approach_clearance_mm
    rotation = _build_tool_rotation(approach_direction, pedicel_axis)

    pose = GraspCutPose(
        grasp_position_mm=grasp_position,
        pre_grasp_position_mm=pre_grasp_position,
        approach_direction_unit=approach_direction,
        cut_position_mm=cut_position,
        pedicel_axis_unit=pedicel_axis,
        rotation_matrix=rotation,
        fruit_center_mm=sphere.center_mm,
        fruit_radius_mm=sphere.radius_mm,
    )

    if obstacle_points_mm is not None and obstacle_points_mm.size > 0:
        if _collides_with_obstacles(pose, obstacle_points_mm, config):
            return PoseResult(
                status=PoseStatus.SKIP_COLLISION,
                fruit_id=fruit_id,
                pose=pose,
                visible_ratio=visible_ratio,
                axis_confidence=axis.confidence,
                sphere_rmse_mm=sphere.rmse_mm,
                cut_offset_error_mm=offset_error,
            )

    return PoseResult(
        status=PoseStatus.READY,
        fruit_id=fruit_id,
        pose=pose,
        visible_ratio=visible_ratio,
        axis_confidence=axis.confidence,
        sphere_rmse_mm=sphere.rmse_mm,
        cut_offset_error_mm=offset_error,
    )


def _collides_with_obstacles(
    pose: GraspCutPose, obstacle_points_mm: np.ndarray, config: GraspCutPoseConfig
) -> bool:
    """접근 경로(pre-grasp → grasp 선분)와 장애물 간 최소 거리가 안전 마진 미만인지 검사한다.

    두 가지 중요한 제한을 둔다:

    1. **목표 과실 주변 점은 검사에서 제외한다.** 그리퍼가 목표 과실을 감싸 파지하는 것은
       충돌이 아니다. 또한 트러스에서 인접 과실은 서로 닿아 자라므로(중심거리 = 두 반지름의
       합), 이 제외 없이는 정상 과실이 전부 충돌로 판정되어 수확이 전면 차단된다.
    2. 접근 경로는 과실을 관통하지 않고 표면에서 멈춘다(pre-grasp → 중심). 과거 구현은
       그리퍼 위치를 과실 반대편으로 잘못 계산해 경로가 과실을 관통했고, 그 결과 접촉한
       이웃 과실이 100% 충돌로 판정됐다.

    주의: 여기서 사용하는 장애물 포인트는 반드시 실제로 관측된 포인트여야 한다. Point Cloud
    Completion 등으로 "복원된" 형상은 모델이 만들어낸 환각일 수 있어 충돌맵에 넣으면
    위험하다(전처리 문서 3장 리스크 1 참고).
    """
    # 목표 과실 표면 + 여유 안쪽의 점은 목표 과실 자신이거나 파지 대상 영역이므로 제외.
    exclusion_radius = pose.fruit_radius_mm + config.target_fruit_exclusion_margin_mm
    distance_from_fruit = np.linalg.norm(obstacle_points_mm - pose.fruit_center_mm, axis=1)
    relevant = obstacle_points_mm[distance_from_fruit > exclusion_radius]
    if relevant.shape[0] == 0:
        return False

    segment_start = pose.pre_grasp_position_mm
    segment_end = pose.grasp_position_mm
    segment_vector = segment_end - segment_start
    segment_length_squared = float(np.dot(segment_vector, segment_vector))

    if segment_length_squared < 1e-9:
        distances = np.linalg.norm(relevant - segment_start, axis=1)
    else:
        relative = relevant - segment_start
        # 각 점을 선분에 정사영한 매개변수 t를 [0, 1]로 클램프(선분 밖으로 나가지 않도록).
        t = np.clip(relative @ segment_vector / segment_length_squared, 0.0, 1.0)
        closest_points = segment_start + t[:, None] * segment_vector
        distances = np.linalg.norm(relevant - closest_points, axis=1)

    return bool(np.min(distances) < config.collision_safety_margin_mm)
