"""파이프라인 설정 스키마.

전처리 코드 내부에 매직 넘버(임계값, 크기 등)를 직접 박아 넣지 않고, 전부 이 모듈의
Pydantic 모델을 통해 주입받도록 강제한다. 기본값은 PRD v2 / 전처리 파이프라인 v2 문서에
명시된 수치를 그대로 반영했으며, 실제 현장 조사 결과가 확보되면 YAML 설정 파일만 교체하면
된다(코드 수정 불필요).

사용 예:
    >>> from harvest_pipeline.config import PipelineConfig
    >>> config = PipelineConfig.from_yaml("configs/default_pipeline.yaml")
    >>> config = PipelineConfig()  # 문서 기본값으로 즉시 사용 가능
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RipenessStage(StrEnum):
    """표준 6단계 숙성도 (PRD 1.4). 과숙(Over-ripe)은 6단계에 포함되지 않는
    별도의 "수확 제외" 사유이므로 이 enum 밖에 존재한다(EXCEPTION 처리, 4.1절 참고)."""

    GREEN = "green"
    BREAKER = "breaker"
    TURNING = "turning"
    PINK = "pink"
    LIGHT_RED = "light_red"
    RED = "red"


class ExceptionReason(StrEnum):
    """정상 숙성 6단계에 속하지 않는 예외 사유(체크리스트 A-7/A-8, PRD 6장)."""

    OVER_RIPE = "over_ripe"
    CRACKING = "cracking"
    DISCOLORATION = "discoloration"
    MALFORMED_CATFACE = "malformed_catface"


class SensorSyncConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_timestamp_drift_ms: float = Field(
        default=15.0, gt=0,
        description="RGB-D/다분광 프레임 간 허용 최대 시간차(ms). 초과 시 SensorSyncError.",
    )


class ColorCalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    white_balance_reference_percentile: float = Field(
        default=99.0, gt=0, lt=100,
        description="Gray-World 화이트밸런스 보정 시 기준으로 삼을 밝기 백분위수.",
    )
    target_mean_luminance: float = Field(
        default=128.0, gt=0, le=255,
        description="선형 노출/게인 정규화 목표 평균 휘도(0~255 스케일).",
    )
    max_gain: float = Field(
        default=3.0, gt=1.0,
        description="과도한 노이즈 증폭을 막기 위한 게인 보정 상한.",
    )


class HighlightSuppressionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    saturation_max: int = Field(default=60, ge=0, le=255, description="하이라이트 후보의 HSV 채도 상한.")
    value_min: int = Field(default=235, ge=0, le=255, description="하이라이트 후보의 HSV 명도 하한.")
    inpaint_radius_px: int = Field(default=5, gt=0, description="cv2.inpaint 반경(px).")


class ShadowRemovalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    background_blur_kernel_px: int = Field(
        default=51, gt=0, description="배경 조도 추정을 위한 median blur 커널 크기(홀수).",
    )

    @model_validator(mode="after")
    def _kernel_must_be_odd(self) -> Self:
        if self.background_blur_kernel_px % 2 == 0:
            raise ValueError("background_blur_kernel_px는 홀수여야 합니다(cv2.medianBlur 제약).")
        return self


class DepthFilterConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    bilateral_diameter: int = Field(default=5, gt=0, description="cv2.bilateralFilter 이웃 지름(px).")
    bilateral_sigma_color: float = Field(
        default=5.0, gt=0,
        description=(
            "Depth 값 차이에 대한 시그마(mm). **반드시 보존하려는 구조의 깊이 기복보다 작아야 "
            "한다.** 방울토마토 한 개의 전체 깊이 기복은 지름과 같은 15~30mm 수준이므로, 이 "
            "값이 그보다 크면 필터가 과실 표면 전체를 '비슷한 값'으로 보고 평탄화해 구면 곡률이 "
            "사라지고 구 피팅 반지름이 과대 추정된다."
        ),
    )
    bilateral_sigma_space: float = Field(default=5.0, gt=0, description="공간 거리 시그마(px).")
    hole_inpaint_radius_px: int = Field(default=3, gt=0)
    near_clip_mm: float = Field(default=100.0, ge=0, description="이 거리 미만 Depth는 노이즈로 간주해 제거.")
    far_clip_mm: float = Field(default=1200.0, gt=0, description="이 거리 초과 Depth는 배경으로 간주해 제거.")


class PointCloudConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    voxel_size_mm: float = Field(default=2.0, gt=0, description="Voxel Downsampling 격자 크기(mm).")
    outlier_k_neighbors: int = Field(default=12, gt=0, description="Statistical Outlier Removal 이웃 수.")
    outlier_std_ratio: float = Field(default=2.0, gt=0, description="평균 이웃거리 대비 표준편차 배수 임계값.")


class RipenessGateConfig(BaseModel):
    """Stage A: 숙성도 판별 게이트 (PRD 5.1-1항)."""

    model_config = ConfigDict(frozen=True)

    confidence_threshold: float = Field(
        default=0.90, gt=0, le=1.0, description="자동 수확 통과를 위한 최소 신뢰도.",
    )
    max_recapture_retries: int = Field(default=2, ge=0, description="신뢰도 미달 시 최대 재촬영 횟수.")
    target_stages: tuple[RipenessStage, ...] = Field(
        default=(RipenessStage.LIGHT_RED, RipenessStage.RED),
        description="수확 대상 숙성 단계.",
    )
    multi_patch_sample_count: int = Field(
        default=5, ge=1, description="과실 표면 부위별 색상 샘플링 패치 수(A-3 불균일 발색 대응).",
    )
    cracking_edge_density_threshold: float = Field(
        default=0.12, gt=0, description="열과(균열) 판정을 위한 표면 엣지 밀도 임계값(A-7).",
    )
    catface_circularity_min: float = Field(
        default=0.80, gt=0, le=1.0,
        description="기형과(Catface) 판정 기준 — 이 값보다 원형도가 낮으면 기형과로 분류(A-8).",
    )


class GraspCutPoseConfig(BaseModel):
    """Stage B: 파지·절단 결합 포즈 (PRD 4장/5.1-3항).

    파지점과 절단점을 독립적으로 추정하지 않는다는 원칙은 유지하되, 둘의 관계를 다음과 같이
    **측면 접근(side approach)** 기구학으로 모델링한다:

    - 그리퍼는 과실 중심을 감싸 파지한다(TCP 목표 = 과실 중심).
    - 절단날은 그리퍼로부터 Pedicel 축 방향으로 `blade_axial_offset_mm`만큼 앞서 있다.
      따라서 절단점은 TCP 위치에 이 오프셋을 더한 곳으로 **유도**되며 별도 추정 대상이 아니다.
    - 로봇은 Pedicel 축에 수직인 방향(카메라 쪽)에서 진입한다. 축과 평행하게 진입하면
      그리퍼가 과실을 관통해야 칼날이 Pedicel에 닿으므로 물리적으로 불가능하다.
    """

    model_config = ConfigDict(frozen=True)

    fruit_diameter_range_mm: tuple[float, float] = Field(
        default=(15.0, 30.0), description="정상 과실 **지름** 허용 범위(품종 편차 포함, PRD 1.4).",
    )
    min_points_for_sphere_fit: int = Field(
        default=30, gt=3, description="구 피팅에 필요한 최소 포인트 수(부족 시 스킵).",
    )
    pedicel_cut_window_mm: tuple[float, float] = Field(
        default=(5.0, 10.0), description="과실 표면 기준 Pedicel 절단 허용창(PRD 1.4).",
    )
    cut_precision_target_mm: float = Field(
        default=2.0, gt=0, description="목표 절단 위치 정밀도(±mm, PRD v2 3장 KPI).",
    )
    occlusion_visible_ratio_min: float = Field(
        default=0.5, gt=0, le=1.0,
        description="이 비율 미만으로 보이는 과실은 포즈 추정을 시도하지 않고 스킵(3장 리스크 대응).",
    )
    min_axis_confidence: float = Field(
        default=0.55, gt=0, le=1.0,
        description=(
            "Pedicel 축 추정 신뢰도 하한. 이 값 미만이면 축 방향이 불확실하다고 보고 스킵한다 "
            "— 잘못된 축으로 절단을 시도하면 과실 어깨나 모주(줄기)를 손상시킬 수 있다."
        ),
    )
    blade_axial_offset_mm: float = Field(
        default=18.0, gt=0,
        description=(
            "그리퍼 TCP에서 절단날까지의 Pedicel 축 방향 거리(mm) — 하드웨어 제작 시 확정되는 상수. "
            "절단점은 과실 표면에서 절단창(5~10mm) 안에 있어야 하므로, 이 값은 "
            "`과실 반지름 + 절단창` 범위와 물리적으로 호환되어야 한다. 지름 15~30mm 과실에서 "
            "유효 범위는 약 12.5~25mm이며, 벗어나면 해당 과실은 하드웨어로 도달 불가로 스킵된다."
        ),
    )
    blade_offset_tolerance_mm: float = Field(
        default=6.0, gt=0,
        description=(
            "고정 오프셋과 이상적 절단 거리(반지름+절단창 중앙)의 허용 불일치(mm). 이 범위를 "
            "넘으면 칼날이 절단창을 벗어나므로 SKIP_UNREACHABLE_OFFSET으로 스킵한다."
        ),
    )
    approach_clearance_mm: float = Field(
        default=60.0, gt=0,
        description="사전 접근(pre-grasp) 지점까지의 후퇴 거리(mm). 이 구간이 충돌 검사 대상이다.",
    )
    collision_safety_margin_mm: float = Field(
        default=5.0, gt=0,
        description=(
            "접근 경로와 장애물 간 최소 이격 거리(mm). 그리퍼 손가락의 물리적 반폭을 기준으로 "
            "설정한다. 과실 지름(15~30mm)의 절반보다 크게 잡으면 트러스에서 서로 접촉해 자라는 "
            "정상 인접 과실이 전부 충돌로 판정되어 수확이 전면 차단된다."
        ),
    )
    target_fruit_exclusion_margin_mm: float = Field(
        default=2.0, ge=0,
        description=(
            "충돌 검사에서 제외할 목표 과실 주변 여유(mm). 그리퍼가 목표 과실을 감싸는 것은 "
            "충돌이 아니므로, 목표 과실 표면 + 이 여유 안쪽의 장애물 점은 검사에서 제외한다."
        ),
    )


class RobotWorkspaceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    reach_mm: float = Field(default=900.0, gt=0, description="로봇팔 리치(PRD v2 4장, 900mm급).")
    truss_height_range_mm: tuple[float, float] = Field(default=(500.0, 2200.0))
    aisle_width_range_mm: tuple[float, float] = Field(default=(800.0, 1200.0))


class TimeoutConfig(BaseModel):
    """PRD 5.1-7항 타임아웃 규칙."""

    model_config = ConfigDict(frozen=True)

    vision_inference_timeout_s: float = Field(default=5.0, gt=0)
    grasp_cut_execution_timeout_s: float = Field(default=20.0, gt=0)
    truss_processing_timeout_s: float = Field(default=480.0, gt=0, description="트러스 전체 처리 타임아웃(8분).")


class LoggingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    log_dir: Path = Field(default=Path("logs"))
    level: str = Field(default="INFO")
    max_bytes: int = Field(default=10_000_000, gt=0, description="로그 파일 로테이션 크기(byte).")
    backup_count: int = Field(default=5, ge=0)
    json_lines_filename: str = Field(default="harvest_pipeline.jsonl")


class PipelineConfig(BaseModel):
    """전체 파이프라인 설정의 최상위 컨테이너."""

    model_config = ConfigDict(frozen=True)

    sensor_sync: SensorSyncConfig = Field(default_factory=SensorSyncConfig)
    color_calibration: ColorCalibrationConfig = Field(default_factory=ColorCalibrationConfig)
    highlight_suppression: HighlightSuppressionConfig = Field(default_factory=HighlightSuppressionConfig)
    shadow_removal: ShadowRemovalConfig = Field(default_factory=ShadowRemovalConfig)
    depth_filter: DepthFilterConfig = Field(default_factory=DepthFilterConfig)
    point_cloud: PointCloudConfig = Field(default_factory=PointCloudConfig)
    ripeness_gate: RipenessGateConfig = Field(default_factory=RipenessGateConfig)
    grasp_cut_pose: GraspCutPoseConfig = Field(default_factory=GraspCutPoseConfig)
    robot_workspace: RobotWorkspaceConfig = Field(default_factory=RobotWorkspaceConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """YAML 설정 파일을 로드해 검증된 PipelineConfig를 반환한다.

        Raises:
            FileNotFoundError: 설정 파일이 존재하지 않는 경우.
            pydantic.ValidationError: 설정값이 스키마 제약을 위반하는 경우.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        """현재 설정을 YAML로 직렬화한다(현장 조사 이후 값 재조정 결과 저장용)."""
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, allow_unicode=True, sort_keys=False)
