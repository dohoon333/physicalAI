"""Stage 0: 공통 전처리.

숙성도 판별(Stage A)과 파지·절단 결합 포즈 추정(Stage B) 양쪽이 공유하는 선행 처리 단계다
(전처리 파이프라인 문서 2장 — v1의 "Stage A/B 병렬 분기" 구조는 세그멘테이션 마스크를
Stage B에서만 생성해 Stage A가 근거 없이 부위별 샘플링을 수행하는 모순이 있어, v2에서
세그멘테이션을 이 공통 단계로 옮겼다).

각 함수는 순수 함수에 가깝게 설계했다(부작용 없이 입력을 받아 새 배열을 반환) — 파이프라인
오케스트레이터(pipeline.py)가 각 단계를 독립적으로 재시도/스킵/로깅할 수 있게 하기 위함이다.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from harvest_pipeline.config import (
    ColorCalibrationConfig,
    DepthFilterConfig,
    HighlightSuppressionConfig,
    PointCloudConfig,
    SensorSyncConfig,
    ShadowRemovalConfig,
)
from harvest_pipeline.exceptions import InvalidImageError, InvalidPointCloudError, SensorSyncError

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - 방어적 폴백, CI/데모 환경엔 항상 설치되어 있음
    _CV2_AVAILABLE = False


def _require_cv2() -> None:
    if not _CV2_AVAILABLE:
        raise ImportError(
            "이 기능은 opencv-python(-headless)이 필요합니다. "
            "`pip install opencv-python-headless`로 설치하세요."
        )


def _validate_hwc_uint8(image: np.ndarray, *, name: str) -> None:
    if image.dtype != np.uint8:
        raise InvalidImageError(f"{name}는 uint8이어야 합니다. 실제 dtype={image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(f"{name}는 (H, W, 3) 형태여야 합니다. 실제 shape={image.shape}")


def check_sensor_sync(
    rgb_timestamp_ms: float, depth_timestamp_ms: float, config: SensorSyncConfig
) -> None:
    """RGB/Depth 프레임 타임스탬프 드리프트를 검사한다.

    Raises:
        SensorSyncError: 허용 드리프트를 초과한 경우. 이는 진짜 하드웨어/동기화 결함이므로
            (미숙과 스킵 같은 정상 분기가 아니라) 예외로 표현한다.
    """
    drift = abs(rgb_timestamp_ms - depth_timestamp_ms)
    if drift > config.max_timestamp_drift_ms:
        raise SensorSyncError(
            f"RGB-Depth 타임스탬프 드리프트({drift:.2f}ms)가 허용치"
            f"({config.max_timestamp_drift_ms}ms)를 초과했습니다."
        )


def calibrate_color(
    rgb: np.ndarray,
    config: ColorCalibrationConfig,
    *,
    white_reference_rgb: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """화이트 레퍼런스 기반 선형 화이트밸런스 + 균일 노출 정규화.

    Args:
        rgb: 입력 이미지.
        config: 캘리브레이션 설정.
        white_reference_rgb: 캘리브레이션 시 촬영한 ColorChecker 흰색/회색 패치의 측정
            RGB값. 제공되면 채널별 게인으로 광원 색온도를 보정한다. **None이면 채널별
            보정을 건너뛰고 전 채널에 동일한 스칼라만 곱하는 노출 정규화만 수행한다.**

    화이트 레퍼런스가 없을 때 Gray-World(장면 전체 평균을 회색으로 맞추는) 방식을 쓰지 않는
    이유: 온실 장면은 잎의 녹색이 화면 대부분을 차지해 "평균이 회색"이라는 가정이 근본적으로
    성립하지 않는다. 이 상태에서 Gray-World를 적용하면 R·B 채널이 과도하게 증폭되어 붉은
    과실의 Hue가 청록 쪽으로 이동하고, Stage A의 색상 기반 숙성도 판별이 완전히 무너진다
    (실제로 붉은 과실이 Green 단계로 오분류되는 문제가 확인되었다). 잘못된 보정보다 보정을
    하지 않는 것이 안전하므로, 채널별 보정은 명시적 레퍼런스가 있을 때만 수행한다.

    비선형 대비 확장(CLAHE 등)을 쓰지 않는 이유는 전처리 문서 3.4절과 동일하다 —
    절대 색도를 보존해야 Hue 임계값 기반 판별이 성립한다.
    """
    _validate_hwc_uint8(rgb, name="rgb")

    rgb_f = rgb.astype(np.float32)

    if white_reference_rgb is not None:
        reference = np.clip(np.asarray(white_reference_rgb, dtype=np.float32), 1e-6, None)
        # 흰색 패치는 세 채널이 동일해야 하므로, 평균 대비 각 채널의 편차를 게인으로 상쇄한다.
        channel_gains = np.clip(
            float(reference.mean()) / reference, 1.0 / config.max_gain, config.max_gain
        )
        rgb_f = rgb_f * channel_gains

    # 노출 정규화: 전 채널에 동일한 스칼라를 곱하므로 채널 간 비율(=색상)이 보존된다.
    # 밝은 영역(상위 백분위)을 기준으로 삼아, 어두운 배경이 넓은 장면에서 과도하게
    # 증폭되는 것을 방지한다.
    percentile = config.white_balance_reference_percentile
    reference_luminance = float(np.percentile(rgb_f.mean(axis=2), percentile))

    if reference_luminance > 1e-6:
        desired_gain = config.target_mean_luminance / reference_luminance

        # 포화(클리핑) 방지 상한. 게인을 곱한 뒤 특정 채널이 255를 넘어 잘리면, 그 채널만
        # 값이 고정되고 나머지 채널은 계속 커져 채널 간 비율이 깨진다. 즉 "전 채널 동일
        # 스칼라라서 색상이 보존된다"는 전제가 클리핑 때문에 무너진다(실제로 붉은 과실의
        # R 채널이 255로 포화되면서 Hue가 14°→34°로 밀리는 문제가 확인되었다).
        #
        # 상한 계산에 퍼센타일이 아니라 **실제 최댓값**을 쓰는 이유: 과실은 화면의 1% 미만을
        # 차지하는 소수 픽셀이라, 99퍼센타일 같은 통계값은 잎 배경 밝기에 지배되어 정작
        # 보호해야 할 과실 픽셀의 클리핑을 막지 못한다. 최댓값 기준이면 클리핑이 원천적으로
        # 발생하지 않아 Hue가 정확히 보존된다(이미 255인 픽셀이 있으면 게인 1.0으로 수렴하는
        # 보수적 동작이 되며, 이는 잘못된 보정보다 안전하다).
        brightest_channel_value = float(rgb_f.max())
        clipping_safe_gain = (
            255.0 / brightest_channel_value if brightest_channel_value > 1e-6 else config.max_gain
        )

        exposure_gain = float(
            np.clip(
                min(desired_gain, clipping_safe_gain),
                1.0 / config.max_gain,
                config.max_gain,
            )
        )
        rgb_f = rgb_f * exposure_gain

    return np.clip(rgb_f, 0, 255).astype(np.uint8)


def suppress_specular_highlight(rgb: np.ndarray, config: HighlightSuppressionConfig) -> np.ndarray:
    """정반사 하이라이트 영역을 검출해 인페인팅으로 채운다(광택 표면·물방울 대응).

    체크리스트 A-6(표면 광택), B-6(관수 방식으로 인한 물방울)에서 지적된 하이라이트
    노이즈를 억제한다.
    """
    _require_cv2()
    _validate_hwc_uint8(rgb, name="rgb")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    highlight_mask = (
        (hsv[:, :, 1] <= config.saturation_max) & (hsv[:, :, 2] >= config.value_min)
    ).astype(np.uint8) * 255

    if not np.any(highlight_mask):
        return rgb

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(bgr, highlight_mask, config.inpaint_radius_px, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


def remove_shadow(rgb: np.ndarray, config: ShadowRemovalConfig) -> np.ndarray:
    """HSV 명도(V) 채널만 조도 정규화해 그림자를 제거한다.

    잎에 의한 불규칙 그림자가 미숙과의 어두운 색상과 혼동되는 것을 방지한다
    (체크리스트 C-4, 전처리 문서 1장).

    RGB 채널별로 각각 나눗셈 정규화하지 않는 이유: 그 방식은 균일한 색상 영역에서
    (채널 값) / (같은 채널의 블러값) ≈ 1 이 되어 모든 채널이 동일한 상수로 수렴하고,
    결과적으로 색상 정보가 회색으로 소실된다(실제로 붉은 과실이 회색으로 변하는 문제가
    확인되었다). 그림자는 본질적으로 "밝기"의 변화이므로 명도 채널만 정규화하고 색상(H)과
    채도(S)는 원본 그대로 보존해야 하며, 이는 Stage A의 Hue 기반 숙성도 판별의 전제 조건이다.
    """
    _require_cv2()
    _validate_hwc_uint8(rgb, name="rgb")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]

    illumination = cv2.medianBlur(value, config.background_blur_kernel_px)
    illumination_safe = np.clip(illumination.astype(np.float32), 1.0, None)

    normalized_value = (value.astype(np.float32) / illumination_safe) * 128.0
    hsv[:, :, 2] = np.clip(normalized_value, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def fill_depth_and_denoise(depth_mm: np.ndarray, config: DepthFilterConfig) -> np.ndarray:
    """Depth 결측치(0)를 인페인팅으로 채우고, 경계를 보존하는 Bilateral Filter로 노이즈를 줄인다.

    근/원거리 클리핑(near/far clip)으로 통로 바닥이나 반대편 트러스 같은 명백한 배경도
    함께 제거해, 이후 세그멘테이션이 관심 영역에만 집중하도록 돕는다.

    인페인팅은 hole(0) 위치에만 영향을 주고, 유효 화소는 원본 float32 값을 비트 단위로
    그대로 보존한다 — 방울토마토 지름(15~30mm)은 far_clip 기준 uint8 양자화 스텝(~수 mm)
    몇 단계로만 표현되면 구면 곡률이 소멸해 구 피팅이 붕괴하므로, 유효 화소를 양자화 왕복에
    노출시켜서는 안 된다.
    """
    _require_cv2()
    if depth_mm.ndim != 2:
        raise InvalidImageError(f"depth_mm은 (H, W) 형태여야 합니다. 실제 shape={depth_mm.shape}")

    depth = depth_mm.astype(np.float32).copy()
    depth[(depth < config.near_clip_mm) | (depth > config.far_clip_mm)] = 0.0

    hole_mask = (depth == 0).astype(np.uint8) * 255
    if np.any(hole_mask):
        valid_mask = depth > 0
        if np.any(valid_mask):
            depth_min = float(depth[valid_mask].min())
            depth_max = float(depth.max())
            depth_range = depth_max - depth_min

            if depth_range > 1e-6:
                # cv2.inpaint는 8bit 단일 채널만 지원하므로 유효 화소를 [depth_min, depth_max] ->
                # [0, 255]로 정규화해 인페인팅한 뒤 **동일한 정변환의 역함수**로 mm 단위로
                # 되돌린다. (기존 결함: cv2.normalize는 배열 전체 min=0 기준 offset 없이
                # 스케일링하는데, 역변환은 depth_min을 더해 정/역변환 식이 어긋나 있었다 —
                # 예: D_min=400, D_max=900일 때 실제 400mm 화소가 621.6mm로 복원되는
                # +221mm 오차. 여기서는 정/역변환에 동일한 depth_min offset을 써서 오차를
                # 없앤다.)
                depth_norm = np.zeros_like(depth, dtype=np.uint8)
                depth_norm[valid_mask] = np.clip(
                    (depth[valid_mask] - depth_min) / depth_range * 255.0, 0, 255
                ).astype(np.uint8)
                inpainted_norm = cv2.inpaint(
                    depth_norm, hole_mask, config.hole_inpaint_radius_px, cv2.INPAINT_NS
                )
                filled_mm = inpainted_norm.astype(np.float32) / 255.0 * depth_range + depth_min

                # inpaint 결과는 hole 위치에만 대입한다 — 유효 화소까지 양자화값으로 덮어쓰면
                # (기존 결함) 위 docstring에서 설명한 구면 곡률 소멸 문제가 발생한다.
                depth = np.where(valid_mask, depth, filled_mm)
            else:
                # 유효 화소가 전부 동일값이면 (depth_max - depth_min) = 0이라 나눗셈이 정의되지
                # 않는다. 채울 근거가 되는 값이 그 하나뿐이므로 인페인팅 없이 바로 대입한다.
                depth = np.where(valid_mask, depth, depth_min)
        # else: 유효 화소가 하나도 없다(전부 클리핑되었거나 원래 0) — 채워 넣을 근거가 되는
        # depth 정보 자체가 없으므로 인페인팅을 건너뛰고 그대로(전부 0) 반환한다. 여기서 예외를
        # 던지지 않는 이유: run_stage0의 호출부(pipeline.process_truss)는 HarvestPipelineError
        # 계열만 잡으므로, 여기서 크래시하면(기존 결함: depth[depth>0].min()의 빈 배열 축소
        # ValueError) 트러스 전체 처리가 격리 없이 죽는다. depth 정보가 없는 프레임은 후속
        # 세그멘테이션이 자연스럽게 과실 0개로 처리하면 되고, 트러스를 중단시킬 필요가 없다.

    denoised = cv2.bilateralFilter(
        depth, config.bilateral_diameter, config.bilateral_sigma_color, config.bilateral_sigma_space
    )
    return denoised


def mask_static_structures(image: np.ndarray, static_structure_mask: np.ndarray) -> np.ndarray:
    """사전 등록된 온실 구조물(지지대·유인끈 등) 마스크 영역을 0으로 지운다.

    캘리브레이션 단계에서 한 번 등록해둔 정적 마스크를 재사용하므로 매 프레임 별도
    연산 비용이 들지 않는다(단순 불리언 마스킹, PRD 6장 리스크 "고정 장애물" 대응).
    """
    if static_structure_mask.shape != image.shape[:2]:
        raise InvalidImageError(
            "static_structure_mask shape이 image의 (H, W)와 일치하지 않습니다: "
            f"{static_structure_mask.shape} vs {image.shape[:2]}"
        )
    masked = image.copy()
    masked[static_structure_mask] = 0
    return masked


def voxel_downsample(points: np.ndarray, config: PointCloudConfig) -> np.ndarray:
    """Voxel 격자 기반 다운샘플링(격자당 평균 좌표 1개만 남김).

    Open3D 같은 무거운 외부 의존성 없이 numpy만으로 벡터화 구현했다 — 파이썬 루프 없이
    numpy의 구조화 배열(unique)을 이용하므로 대용량 포인트클라우드에서도 메모리·연산
    효율적이다("성능 및 메모리" 요구사항).
    """
    if points.size == 0:
        return points
    _validate_point_cloud_shape(points)

    voxel_size = config.voxel_size_mm
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)

    # 3개 정수 좌표를 단일 정수 키로 압축해 np.unique 그룹핑에 사용(파이썬 루프 회피).
    # 좌표가 매우 큰 값(수십만 mm 이상)이면 비트 충돌 가능성이 있으나, 온실 스케일(<10m)
    # 좌표계에서는 안전하다.
    voxel_indices_shifted = voxel_indices - voxel_indices.min(axis=0)
    dims = voxel_indices_shifted.max(axis=0) + 1
    flat_keys = np.ravel_multi_index(voxel_indices_shifted.T, dims.astype(np.int64))

    unique_keys, inverse, counts = np.unique(flat_keys, return_inverse=True, return_counts=True)

    sums = np.zeros((unique_keys.size, 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    return (sums / counts[:, None]).astype(np.float32)


def remove_statistical_outliers(points: np.ndarray, config: PointCloudConfig) -> np.ndarray:
    """이웃 평균 거리가 통계적으로 비정상적인 점을 제거한다(Open3D의 SOR와 동일한 원리).

    cKDTree로 k-최근접 이웃 거리를 벡터화 계산한다(O(N log N) 수준으로, 순수 파이썬
    이중 루프 대비 대용량에서 훨씬 빠르다).
    """
    if points.shape[0] <= config.outlier_k_neighbors:
        return points  # 포인트가 너무 적으면 통계적 판단이 무의미 — 원본 그대로 반환
    _validate_point_cloud_shape(points)

    tree = cKDTree(points)
    # k+1: 첫 번째 이웃은 자기 자신(거리 0)이므로 제외하기 위해 하나 더 조회.
    distances, _ = tree.query(points, k=config.outlier_k_neighbors + 1)
    mean_neighbor_distance = distances[:, 1:].mean(axis=1)

    global_mean = mean_neighbor_distance.mean()
    global_std = mean_neighbor_distance.std()
    threshold = global_mean + config.outlier_std_ratio * global_std

    inlier_mask = mean_neighbor_distance <= threshold
    return points[inlier_mask]


def _validate_point_cloud_shape(points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise InvalidPointCloudError(
            f"포인트클라우드는 (N, 3) 형태여야 합니다. 실제 shape={points.shape}"
        )
