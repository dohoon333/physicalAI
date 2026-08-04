"""전처리 파이프라인 테스트.

테스트가 검증하는 핵심 불변식(invariant)은 개발 중 실제로 발견된 결함들에 대응한다:
- 색상 보정/그림자 제거가 Hue를 보존해야 한다(숙성도 판별의 전제).
- Multi-patch 샘플링이 배경 픽셀을 섞지 않아야 한다.
- 축 신뢰도 지표가 막대형과 원반형을 구분해야 한다.
- 구 피팅에 Pedicel 포인트가 섞이면 안 된다.
- 개별 과실 오류가 트러스 전체 처리를 중단시키지 않아야 한다.

실행:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harvest_pipeline import stage0_common as s0  # noqa: E402
from harvest_pipeline.config import (  # noqa: E402
    ColorCalibrationConfig,
    DepthFilterConfig,
    GraspCutPoseConfig,
    LoggingConfig,
    PipelineConfig,
    PointCloudConfig,
    RipenessGateConfig,
    RipenessStage,
    SensorSyncConfig,
    ShadowRemovalConfig,
)
from harvest_pipeline.exceptions import (  # noqa: E402
    InvalidImageError,
    InvalidPointCloudError,
    SegmentationModelError,
    SensorSyncError,
)
from harvest_pipeline.interfaces import (  # noqa: E402
    ClassicalColorSegmentationModel,
    GeometricPedicelPoseEstimator,
    RuleBasedRipenessClassifier,
    SegmentationResult,
    sample_multi_patch_regions,
)
from harvest_pipeline.pipeline import (  # noqa: E402
    CameraIntrinsics,
    FrameInput,
    FruitOutcome,
    HarvestPreprocessingPipeline,
    deproject_depth_to_points,
)
from harvest_pipeline.stage_a_ripeness import (  # noqa: E402
    GateDecision,
    detect_cracking,
    detect_malformation,
    evaluate_ripeness_gate,
)
from harvest_pipeline.stage_b_grasp_cut import (  # noqa: E402
    PoseStatus,
    compute_grasp_cut_pose,
    fit_sphere_least_squares,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config(tmp_path: Path) -> PipelineConfig:
    """로그를 임시 디렉터리에 쓰는 테스트용 설정."""
    return PipelineConfig(logging=LoggingConfig(log_dir=tmp_path / "logs"))


@pytest.fixture
def red_fruit_image() -> tuple[np.ndarray, np.ndarray]:
    """초록 배경 위 붉은 원형 과실 이미지와 그 마스크."""
    rgb = np.full((160, 160, 3), (40, 95, 35), dtype=np.uint8)
    cv2.circle(rgb, (80, 80), 30, (215, 30, 25), thickness=-1)
    mask = np.zeros((160, 160), dtype=bool)
    cv2.circle(mask.view(np.uint8), (80, 80), 30, 1, thickness=-1)
    return rgb, mask


def make_sphere_points(
    center: tuple[float, float, float], radius_mm: float, count: int = 400, seed: int = 0
) -> np.ndarray:
    """구 표면에 균일 분포한 포인트를 생성한다."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return np.asarray(center) + directions * radius_mm + rng.normal(0, 0.15, (count, 3))


def make_pedicel_points(
    fruit_center: tuple[float, float, float], radius_mm: float, count: int = 60, seed: int = 1
) -> np.ndarray:
    """과실 상단에서 **−Y 방향**으로 뻗은 얇은 막대형 포인트를 생성한다.

    카메라 좌표계는 X 우, Y 하, Z 광축(깊이)이므로 이미지상 "위"는 −Y다. Pedicel을 +Z로
    두면 카메라 뒤쪽(관측 불가 영역)에 놓이는 셈이고, 접근 방향 계산도 광축과 평행해져
    축퇴(degenerate) 분기로 빠진다 — 이 헬퍼의 이전 버전이 그런 비물리적 배치를 만들어
    프로덕션 코드의 좌표계 결함을 가려주고 있었다.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(fruit_center) + np.array([0.0, -radius_mm, 0.0])
    return base + np.column_stack(
        [rng.normal(0, 0.4, count), -rng.uniform(0, 14, count), rng.normal(0, 0.4, count)]
    )


# ---------------------------------------------------------------------------
# 설정 (파라미터화 및 버전 관리)
# ---------------------------------------------------------------------------

class TestConfig:
    def test_yaml_roundtrip_matches_defaults(self, tmp_path: Path) -> None:
        original = PipelineConfig()
        path = tmp_path / "config.yaml"
        original.to_yaml(path)
        assert PipelineConfig.from_yaml(path).model_dump() == original.model_dump()

    def test_shipped_default_yaml_matches_code_defaults(self) -> None:
        """configs/default_pipeline.yaml이 코드 기본값과 동기화되어 있는지 검증한다.

        설정 파일과 코드 기본값이 어긋나면 "문서에 적힌 값"과 "실제 동작"이 달라져
        재현성이 깨지므로, CI에서 상시 감시할 가치가 있는 불변식이다.
        """
        shipped = Path(__file__).resolve().parent.parent / "configs" / "default_pipeline.yaml"
        assert PipelineConfig.from_yaml(shipped).model_dump() == PipelineConfig().model_dump()

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("does_not_exist.yaml")

    def test_invalid_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            RipenessGateConfig(confidence_threshold=1.5)

    def test_even_blur_kernel_rejected(self) -> None:
        """cv2.medianBlur는 홀수 커널만 허용하므로 설정 단계에서 걸러야 한다."""
        with pytest.raises(ValueError):
            ShadowRemovalConfig(background_blur_kernel_px=50)

    def test_config_is_immutable(self) -> None:
        cfg = RipenessGateConfig()
        with pytest.raises(ValueError):
            cfg.confidence_threshold = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------

class TestSensorSync:
    def test_within_tolerance_passes(self) -> None:
        s0.check_sensor_sync(1000.0, 1010.0, SensorSyncConfig(max_timestamp_drift_ms=15.0))

    def test_exceeding_tolerance_raises(self) -> None:
        with pytest.raises(SensorSyncError, match="드리프트"):
            s0.check_sensor_sync(1000.0, 1100.0, SensorSyncConfig(max_timestamp_drift_ms=15.0))


class TestColorCalibration:
    def test_preserves_hue(self) -> None:
        """색상 보정은 Hue를 보존해야 한다 — Stage A가 Hue 임계값으로 판정하기 때문이다."""
        rgb = np.full((60, 60, 3), (40, 95, 35), dtype=np.uint8)
        cv2.circle(rgb, (30, 30), 15, (215, 30, 25), thickness=-1)

        before = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[30, 30, 0]
        after = cv2.cvtColor(
            s0.calibrate_color(rgb, ColorCalibrationConfig()), cv2.COLOR_RGB2HSV
        )[30, 30, 0]

        assert abs(int(after) - int(before)) <= 2  # uint8 양자화 오차 허용

    def test_never_clips_bright_pixels(self) -> None:
        """게인 적용 후에도 채널이 255로 포화되지 않아야 한다(포화는 Hue를 왜곡한다)."""
        rgb = np.full((40, 40, 3), (20, 20, 20), dtype=np.uint8)
        rgb[10:20, 10:20] = (250, 60, 40)  # 이미 매우 밝은 영역

        result = s0.calibrate_color(rgb, ColorCalibrationConfig())
        assert result[15, 15].max() <= 255
        # 원본 비율이 유지되어야 한다.
        original_ratio = 60.0 / 250.0
        result_ratio = float(result[15, 15, 1]) / float(result[15, 15, 0])
        assert abs(result_ratio - original_ratio) < 0.05

    def test_white_reference_neutralizes_color_cast(self) -> None:
        """화이트 레퍼런스를 주면 광원 색편향이 보정되어야 한다."""
        # 청색이 강한 광원 아래 촬영된 회색 표면.
        rgb = np.full((20, 20, 3), (100, 110, 140), dtype=np.uint8)
        result = s0.calibrate_color(
            rgb, ColorCalibrationConfig(), white_reference_rgb=(100.0, 110.0, 140.0)
        )
        channels = result[10, 10].astype(int)
        assert channels.max() - channels.min() <= 12  # 거의 중성 회색으로 수렴

    def test_rejects_non_uint8(self) -> None:
        with pytest.raises(InvalidImageError):
            s0.calibrate_color(np.zeros((10, 10, 3), dtype=np.float32), ColorCalibrationConfig())


class TestShadowRemoval:
    def test_preserves_hue_of_uniform_region(self) -> None:
        """균일한 색상 영역이 회색으로 변하지 않아야 한다.

        RGB 채널별 나눗셈 정규화를 쓰면 균일 영역의 모든 채널이 같은 상수로 수렴해
        색상 정보가 소실되는 결함이 있었다. 명도 채널만 정규화하면 방지된다.
        """
        rgb = np.full((120, 120, 3), (40, 95, 35), dtype=np.uint8)
        cv2.circle(rgb, (60, 60), 35, (215, 30, 25), thickness=-1)

        result = s0.remove_shadow(rgb, ShadowRemovalConfig())
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)

        assert hsv[60, 60, 1] > 100, "채도가 유지되어야 한다(회색화 방지)"
        before_hue = int(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[60, 60, 0])
        assert abs(int(hsv[60, 60, 0]) - before_hue) <= 2

    def test_reduces_illumination_gradient(self) -> None:
        """좌우로 밝기 기울기가 있는 이미지에서 기울기가 완화되어야 한다."""
        gradient = np.linspace(60, 220, 120, dtype=np.float32)
        rgb = np.repeat(gradient[None, :, None], 120, axis=0)
        rgb = np.repeat(rgb, 3, axis=2).astype(np.uint8)

        result = s0.remove_shadow(rgb, ShadowRemovalConfig())
        value_channel = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)[:, :, 2].astype(float)
        before_spread = float(gradient.max() - gradient.min())
        after_spread = float(value_channel.max() - value_channel.min())
        assert after_spread < before_spread


class TestDepthFilter:
    def test_fills_holes(self) -> None:
        depth = np.full((80, 80), 500.0, dtype=np.float32)
        depth[30:40, 30:40] = 0.0  # 결측 영역

        result = s0.fill_depth_and_denoise(depth, DepthFilterConfig())
        assert np.count_nonzero(result == 0) == 0

    def test_clips_out_of_range(self) -> None:
        depth = np.full((40, 40), 500.0, dtype=np.float32)
        depth[0:5, :] = 2000.0  # far clip 초과
        result = s0.fill_depth_and_denoise(
            depth, DepthFilterConfig(near_clip_mm=100.0, far_clip_mm=1200.0)
        )
        assert result[0:5, :].max() < 2000.0

    def test_preserves_sphere_curvature(self) -> None:
        """기본 설정의 bilateral 필터가 과실 구면 곡률을 평탄화하지 않아야 한다.

        sigma_color가 과실의 깊이 기복보다 크면 표면 전체가 평균화되어 구 피팅 반지름이
        과대 추정되는 결함이 있었다.
        """
        size = 60
        center, radius_px = size // 2, 20
        y_grid, x_grid = np.ogrid[:size, :size]
        squared = (x_grid - center) ** 2 + (y_grid - center) ** 2
        depth = np.full((size, size), 600.0, dtype=np.float32)
        bulge = np.sqrt(np.maximum(radius_px**2 - squared, 0))
        inside = squared <= radius_px**2
        depth[inside] = (600.0 - bulge)[inside]

        before_relief = float(depth[inside].max() - depth[inside].min())
        filtered = s0.fill_depth_and_denoise(depth, DepthFilterConfig())
        after_relief = float(filtered[inside].max() - filtered[inside].min())

        assert after_relief > before_relief * 0.85, "구면 기복의 85% 이상이 보존되어야 한다"


class TestPointCloudOps:
    def test_voxel_downsample_reduces_count(self) -> None:
        rng = np.random.default_rng(0)
        points = rng.normal(0, 20, (5000, 3)).astype(np.float32)
        result = s0.voxel_downsample(points, PointCloudConfig(voxel_size_mm=5.0))
        assert 0 < result.shape[0] < points.shape[0]
        assert result.shape[1] == 3

    def test_voxel_downsample_handles_empty(self) -> None:
        empty = np.empty((0, 3), dtype=np.float32)
        assert s0.voxel_downsample(empty, PointCloudConfig()).shape[0] == 0

    def test_outlier_removal_drops_far_points(self) -> None:
        rng = np.random.default_rng(0)
        cluster = rng.normal(0, 1.0, (300, 3))
        outliers = np.array([[500.0, 500.0, 500.0], [-400.0, 0.0, 0.0]])
        points = np.vstack([cluster, outliers]).astype(np.float32)

        result = s0.remove_statistical_outliers(points, PointCloudConfig())
        assert result.shape[0] < points.shape[0]
        assert np.max(np.abs(result)) < 100.0

    def test_outlier_removal_skips_tiny_cloud(self) -> None:
        points = np.zeros((5, 3), dtype=np.float32)
        result = s0.remove_statistical_outliers(points, PointCloudConfig(outlier_k_neighbors=12))
        assert result.shape[0] == 5

    def test_rejects_bad_shape(self) -> None:
        with pytest.raises(InvalidPointCloudError):
            s0.voxel_downsample(np.zeros((10, 2), dtype=np.float32), PointCloudConfig())


class TestStructureMasking:
    def test_masks_region(self) -> None:
        image = np.full((20, 20, 3), 100, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=bool)
        mask[0:5, :] = True
        result = s0.mask_static_structures(image, mask)
        assert result[0:5].sum() == 0
        assert result[5:].sum() > 0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(InvalidImageError):
            s0.mask_static_structures(
                np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((5, 5), dtype=bool)
            )


# ---------------------------------------------------------------------------
# Segmentation / Multi-patch
# ---------------------------------------------------------------------------

class TestSegmentation:
    def test_detects_fruit_and_separates_pedicel(self) -> None:
        """열림 연산으로 얇은 Pedicel이 본체 마스크에서 분리되어야 한다."""
        rgb = np.full((160, 160, 3), (40, 95, 35), dtype=np.uint8)
        cv2.circle(rgb, (80, 90), 22, (215, 30, 25), thickness=-1)
        cv2.rectangle(rgb, (77, 45), (83, 68), (215, 30, 25), thickness=-1)  # Pedicel

        instances = ClassicalColorSegmentationModel(
            min_instance_area_px=200, pedicel_removal_kernel_px=11
        ).predict(rgb).fruit_instances

        assert len(instances) == 1
        instance = instances[0]
        assert instance.pedicel_mask is not None
        assert np.count_nonzero(instance.pedicel_mask) > 0
        # 본체 마스크에는 Pedicel 영역이 포함되지 않아야 한다.
        assert not instance.mask[50, 80]
        assert instance.mask[90, 80]
        # combined_mask는 둘을 모두 포함한다.
        assert instance.combined_mask[50, 80]

    def test_ignores_small_noise(self) -> None:
        rgb = np.full((100, 100, 3), (40, 95, 35), dtype=np.uint8)
        cv2.circle(rgb, (50, 50), 3, (215, 30, 25), thickness=-1)
        result = ClassicalColorSegmentationModel(min_instance_area_px=500).predict(rgb)
        assert len(result.fruit_instances) == 0

    def test_empty_result_is_valid(self) -> None:
        assert SegmentationResult().fruit_instances == ()


class TestMultiPatchSampling:
    def test_excludes_background_pixels(self) -> None:
        """패치에 배경(잎) 픽셀이 섞이면 Hue 평균이 오염되어 오판별을 유발한다.

        bounding box를 직사각형으로 잘라내던 초기 구현은 구형 과실의 모서리에서 배경을
        포함해, 완전히 붉은 과실이 light_red로 잘못 분류되는 결함이 있었다.
        """
        rgb = np.full((120, 120, 3), (40, 95, 35), dtype=np.uint8)  # 초록 배경
        mask = np.zeros((120, 120), dtype=bool)
        cv2.circle(rgb, (60, 60), 30, (215, 30, 25), thickness=-1)
        cv2.circle(mask.view(np.uint8), (60, 60), 30, 1, thickness=-1)

        patches = sample_multi_patch_regions(rgb, mask, RipenessGateConfig())
        assert len(patches) > 0
        for patch in patches:
            unique_colors = np.unique(patch.reshape(-1, 3), axis=0)
            for color in unique_colors:
                assert not np.array_equal(color, [40, 95, 35]), "배경색이 패치에 포함됨"

    def test_returns_empty_for_empty_mask(self) -> None:
        rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        assert sample_multi_patch_regions(rgb, np.zeros((50, 50), dtype=bool), RipenessGateConfig()) == []

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(InvalidImageError):
            sample_multi_patch_regions(
                np.zeros((50, 50, 3), dtype=np.uint8),
                np.zeros((10, 10), dtype=bool),
                RipenessGateConfig(),
            )


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------

class TestRipenessClassifier:
    @pytest.mark.parametrize(
        ("rgb_color", "expected"),
        [
            ((215, 30, 25), RipenessStage.RED),
            ((226, 148, 38), RipenessStage.BREAKER),
            ((90, 170, 60), RipenessStage.GREEN),
        ],
    )
    def test_classifies_by_hue(
        self, rgb_color: tuple[int, int, int], expected: RipenessStage
    ) -> None:
        patch = np.full((10, 1, 3), rgb_color, dtype=np.uint8)
        assert RuleBasedRipenessClassifier().predict([patch]).stage is expected

    def test_empty_patches_raises(self) -> None:
        with pytest.raises(InvalidImageError):
            RuleBasedRipenessClassifier().predict([])


class TestRipenessGate:
    def test_ripe_fruit_proceeds(self, red_fruit_image) -> None:
        rgb, _ = red_fruit_image
        instance = ClassicalColorSegmentationModel(min_instance_area_px=200).predict(
            rgb
        ).fruit_instances[0]

        result = evaluate_ripeness_gate(
            rgb, instance, RuleBasedRipenessClassifier(), RipenessGateConfig(), fruit_id="F1"
        )
        assert result.decision is GateDecision.HARVEST
        assert result.should_proceed_to_stage_b
        assert result.ripeness_stage is RipenessStage.RED

    def test_immature_fruit_is_skipped(self) -> None:
        rgb = np.full((160, 160, 3), (20, 20, 20), dtype=np.uint8)
        cv2.circle(rgb, (80, 80), 30, (226, 148, 38), thickness=-1)  # breaker 단계
        instance = ClassicalColorSegmentationModel(min_instance_area_px=200).predict(
            rgb
        ).fruit_instances[0]

        result = evaluate_ripeness_gate(
            rgb, instance, RuleBasedRipenessClassifier(), RipenessGateConfig(), fruit_id="F1"
        )
        assert result.decision is GateDecision.SKIP_IMMATURE
        assert not result.should_proceed_to_stage_b

    def test_cracked_fruit_is_excluded(self) -> None:
        rgb = np.full((160, 160, 3), (40, 95, 35), dtype=np.uint8)
        cv2.circle(rgb, (80, 80), 32, (215, 30, 25), thickness=-1)
        for offset in range(-24, 25, 5):  # 균열선
            cv2.line(rgb, (80 + offset, 55), (80 + offset - 7, 105), (55, 8, 8), thickness=2)

        instance = ClassicalColorSegmentationModel(min_instance_area_px=200).predict(
            rgb
        ).fruit_instances[0]
        result = evaluate_ripeness_gate(
            rgb, instance, RuleBasedRipenessClassifier(), RipenessGateConfig(), fruit_id="F1"
        )
        assert result.decision is GateDecision.SKIP_EXCEPTION

    def test_recapture_is_attempted_on_low_confidence(self, red_fruit_image) -> None:
        """신뢰도 미달 시 설정된 횟수만큼 재촬영을 시도해야 한다."""
        rgb, _ = red_fruit_image
        instance = ClassicalColorSegmentationModel(min_instance_area_px=200).predict(
            rgb
        ).fruit_instances[0]

        call_count = 0

        def recapture() -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return rgb

        class NeverConfidentClassifier:
            def predict(self, patches_rgb):  # type: ignore[no-untyped-def]
                return RuleBasedRipenessClassifier().predict(patches_rgb).__class__(
                    stage=RipenessStage.RED, confidence=0.1
                )

        result = evaluate_ripeness_gate(
            rgb,
            instance,
            NeverConfidentClassifier(),
            RipenessGateConfig(max_recapture_retries=2),
            fruit_id="F1",
            recapture_fn=recapture,
        )
        assert result.decision is GateDecision.SKIP_LOW_CONFIDENCE
        assert call_count == 2

    def test_malformation_detected_on_irregular_shape(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        # 별 모양처럼 둘레가 긴 형상 → 원형도 낮음
        points = np.array(
            [[50, 10], [60, 40], [90, 50], [60, 60], [50, 90], [40, 60], [10, 50], [40, 40]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask.view(np.uint8), [points], 1)
        assert detect_malformation(mask, RipenessGateConfig()) is True

    def test_circle_is_not_malformed(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        cv2.circle(mask.view(np.uint8), (50, 50), 30, 1, thickness=-1)
        assert detect_malformation(mask, RipenessGateConfig()) is False

    def test_clean_fruit_has_no_cracking(self, red_fruit_image) -> None:
        rgb, mask = red_fruit_image
        assert detect_cracking(rgb, mask, RipenessGateConfig()) is False


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------

class TestSphereFit:
    def test_recovers_known_sphere(self) -> None:
        points = make_sphere_points((100.0, 50.0, 600.0), 11.0)
        result = fit_sphere_least_squares(points)
        assert abs(result.radius_mm - 11.0) < 0.5
        assert np.allclose(result.center_mm, [100.0, 50.0, 600.0], atol=0.5)
        assert result.rmse_mm < 1.0

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(InvalidPointCloudError):
            fit_sphere_least_squares(np.zeros((3, 3), dtype=np.float32))

    def test_bad_shape_raises(self) -> None:
        with pytest.raises(InvalidPointCloudError):
            fit_sphere_least_squares(np.zeros((10, 2), dtype=np.float32))


class TestPedicelAxisEstimator:
    def test_rod_has_high_linearity(self) -> None:
        """막대형(Pedicel) 포인트는 높은 linearity를 가져야 한다."""
        points = make_pedicel_points((0.0, 0.0, 0.0), 0.0)
        estimate = GeometricPedicelPoseEstimator().estimate(points)
        assert estimate.confidence > 0.55
        # 카메라 좌표계에서 Pedicel은 이미지상 위쪽(−Y)으로 뻗으므로 주축은 Y축과 정렬된다.
        assert abs(abs(estimate.direction_unit[1]) - 1.0) < 0.1

    def test_disc_has_low_linearity(self) -> None:
        """원반형(Pedicel 없는 구면 캡) 포인트는 낮은 linearity를 가져야 한다.

        '최대 고유값 / 전체 분산' 지표는 이 둘을 구분하지 못해 잘못된 측면 축을 통과시켰다.
        """
        rng = np.random.default_rng(3)
        points = np.column_stack(
            [rng.normal(0, 10, 300), rng.normal(0, 10, 300), rng.normal(0, 0.5, 300)]
        )
        assert GeometricPedicelPoseEstimator().estimate(points).confidence < 0.3

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(InvalidPointCloudError):
            GeometricPedicelPoseEstimator().estimate(np.zeros((2, 3), dtype=np.float32))


class TestGraspCutPose:
    def test_produces_ready_pose(self) -> None:
        center = (100.0, 50.0, 600.0)
        radius = 11.0
        result = compute_grasp_cut_pose(
            make_sphere_points(center, radius),
            GraspCutPoseConfig(),
            GeometricPedicelPoseEstimator(),
            fruit_id="F1",
            visible_ratio=0.9,
            pedicel_points_mm=make_pedicel_points(center, radius),
        )
        assert result.status is PoseStatus.READY
        assert result.is_executable
        assert result.pose is not None
        # 절단 위치는 과실 표면 밖, 절단창(5~10mm) 범위에 있어야 한다.
        distance = float(np.linalg.norm(result.pose.cut_position_mm - np.asarray(center)))
        assert radius + 5.0 <= distance <= radius + 10.0

    def test_pedicel_points_must_not_inflate_radius(self) -> None:
        """Pedicel 포인트를 본체에 섞으면 반지름이 과대 추정된다(회귀 방지).

        구 피팅에 Pedicel을 포함시켜 12mm 과실이 17mm로 추정되던 결함에 대응한다.
        """
        center = (0.0, 0.0, 500.0)
        radius = 11.0
        body = make_sphere_points(center, radius)
        pedicel = make_pedicel_points(center, radius)

        correct = fit_sphere_least_squares(body).radius_mm
        contaminated = fit_sphere_least_squares(np.vstack([body, pedicel])).radius_mm

        assert abs(correct - radius) < 0.5
        assert contaminated > correct + 1.0, "오염된 입력은 반지름을 부풀린다"

    def test_occluded_fruit_is_skipped(self) -> None:
        center = (0.0, 0.0, 500.0)
        result = compute_grasp_cut_pose(
            make_sphere_points(center, 11.0),
            GraspCutPoseConfig(occlusion_visible_ratio_min=0.5),
            GeometricPedicelPoseEstimator(),
            fruit_id="F1",
            visible_ratio=0.2,
        )
        assert result.status is PoseStatus.SKIP_OCCLUDED

    def test_oversize_fruit_is_skipped(self) -> None:
        center = (0.0, 0.0, 500.0)
        result = compute_grasp_cut_pose(
            make_sphere_points(center, 40.0),
            GraspCutPoseConfig(),
            GeometricPedicelPoseEstimator(),
            fruit_id="F1",
            visible_ratio=0.9,
            pedicel_points_mm=make_pedicel_points(center, 40.0),
        )
        assert result.status is PoseStatus.SKIP_SIZE_OUT_OF_RANGE

    def test_missing_pedicel_is_skipped(self) -> None:
        """Pedicel 포인트가 없으면 READY가 될 수 없다(회귀 방지).

        과거 구현은 과실 본체에서 "중심보다 Z가 큰 상단 절반"을 뽑아 Pedicel 축을 대체
        추정했다. 그런데 카메라 좌표계에서 +Z는 "카메라에서 먼 쪽"이고 Depth 센서는 과실의
        대향면만 관측하므로 그 조건을 만족하는 점은 원리적으로 0개였다. 즉 fallback이 항상
        실패하는 죽은 경로였고, 그 사실이 SKIP_INSUFFICIENT_POINTS로 위장돼 있었다.
        """
        result = compute_grasp_cut_pose(
            make_sphere_points((0.0, 0.0, 500.0), 11.0),
            GraspCutPoseConfig(),
            GeometricPedicelPoseEstimator(),
            fruit_id="F1",
            visible_ratio=0.9,
        )
        assert result.status is PoseStatus.SKIP_NO_PEDICEL
        assert not result.is_executable

    def test_low_axis_confidence_is_skipped(self) -> None:
        """Pedicel 포인트가 막대형이 아니라 납작한 원반형이면 축 신뢰도 부족으로 스킵된다."""
        rng = np.random.default_rng(11)
        center = (0.0, 0.0, 500.0)
        # x-y 평면에 퍼진 원반형 점군 — linearity가 낮아 축 방향을 신뢰할 수 없다.
        disc_points = np.asarray(center) + np.column_stack(
            [rng.normal(0, 8.0, 200), rng.normal(0, 8.0, 200), rng.normal(0, 0.3, 200)]
        )
        result = compute_grasp_cut_pose(
            make_sphere_points(center, 11.0),
            GraspCutPoseConfig(),
            GeometricPedicelPoseEstimator(),
            fruit_id="F1",
            visible_ratio=0.9,
            pedicel_points_mm=disc_points,
        )
        assert result.status is PoseStatus.SKIP_LOW_AXIS_CONFIDENCE

    def test_collision_on_approach_path_is_detected(self) -> None:
        """pre-grasp → grasp 접근 경로를 막는 장애물은 SKIP_COLLISION이어야 한다."""
        center = (0.0, 0.0, 500.0)
        radius = 11.0
        config = GraspCutPoseConfig()

        clean = compute_grasp_cut_pose(
            make_sphere_points(center, radius),
            config,
            GeometricPedicelPoseEstimator(),
            fruit_id="F1",
            visible_ratio=0.9,
            pedicel_points_mm=make_pedicel_points(center, radius),
        )
        assert clean.status is PoseStatus.READY
        assert clean.pose is not None

        # 접근 경로의 중간 지점에 장애물을 놓는다(목표 과실 제외 반경 밖).
        midpoint = (clean.pose.pre_grasp_position_mm + clean.pose.grasp_position_mm) / 2.0
        obstacle = midpoint.reshape(1, 3)

        blocked = compute_grasp_cut_pose(
            make_sphere_points(center, radius),
            config,
            GeometricPedicelPoseEstimator(),
            fruit_id="F1",
            visible_ratio=0.9,
            pedicel_points_mm=make_pedicel_points(center, radius),
            obstacle_points_mm=obstacle,
        )
        assert blocked.status is PoseStatus.SKIP_COLLISION

    def test_touching_neighbor_fruit_does_not_block_harvest(self) -> None:
        """트러스에서 서로 접촉해 자라는 정상 인접 과실은 수확을 막지 않아야 한다(회귀 방지).

        과거 구현은 그리퍼 위치를 과실 반대편으로 잘못 계산해 접근 경로가 과실을 관통했고,
        그 결과 중심거리가 두 반지름의 합인(= 표면이 맞닿은) 정상 인접 과실이 100%
        SKIP_COLLISION으로 판정됐다. 방울토마토 밀집 트러스는 과실이 서로 닿아 자라므로
        이 상태에서는 수확 가능 과실이 하나도 산출되지 않는다.
        """
        center = np.array([0.0, 0.0, 500.0])
        radius = 11.0
        config = GraspCutPoseConfig()
        body = make_sphere_points(tuple(center), radius)
        pedicel = make_pedicel_points(tuple(center), radius)

        # 측면 접촉과 Pedicel 축 방향 접촉 둘 다 검증한다.
        for label, offset in [
            ("측면", np.array([2 * radius, 0.0, 0.0])),
            ("축방향(위)", np.array([0.0, -2 * radius, 0.0])),
            ("축방향(아래)", np.array([0.0, 2 * radius, 0.0])),
        ]:
            neighbor = make_sphere_points(tuple(center + offset), radius, seed=5)
            result = compute_grasp_cut_pose(
                body,
                config,
                GeometricPedicelPoseEstimator(),
                fruit_id="F1",
                visible_ratio=0.9,
                pedicel_points_mm=pedicel,
                obstacle_points_mm=neighbor,
            )
            assert result.status is PoseStatus.READY, f"{label} 접촉 이웃이 수확을 차단함"

    def test_pose_geometry_is_physically_valid(self) -> None:
        """산출된 포즈가 물리적으로 실행 가능한 기하를 가져야 한다(회귀 방지).

        과거 구현은 `norm(blade_offset)`으로 오프셋의 방향 정보를 버리고 크기만 Pedicel 축에
        투영해, TCP가 과실 **반대편** 27.5mm 지점에 놓였다. 그리퍼가 과실을 잡을 수 없고
        접근 경로가 과실을 관통하는 위치였다. 또한 회전이 미결정이라 실질 5-DOF였다.
        """
        center = np.array([0.0, 0.0, 500.0])
        radius = 11.0
        config = GraspCutPoseConfig()

        result = compute_grasp_cut_pose(
            make_sphere_points(tuple(center), radius),
            config,
            GeometricPedicelPoseEstimator(),
            fruit_id="F1",
            visible_ratio=0.9,
            pedicel_points_mm=make_pedicel_points(tuple(center), radius),
        )
        assert result.status is PoseStatus.READY
        pose = result.pose
        assert pose is not None

        # TCP는 과실 중심을 목표로 한다(그리퍼가 과실을 감싼다).
        assert np.linalg.norm(pose.grasp_position_mm - pose.fruit_center_mm) < 1e-6

        # 절단점은 과실 표면 바깥, 절단창 근방에 있어야 한다.
        cut_distance = float(np.linalg.norm(pose.cut_position_mm - pose.fruit_center_mm))
        assert cut_distance > pose.fruit_radius_mm, "절단점이 과실 내부에 있음"
        cut_window_min, cut_window_max = config.pedicel_cut_window_mm
        assert cut_distance <= pose.fruit_radius_mm + cut_window_max + config.blade_offset_tolerance_mm

        # 접근 방향은 Pedicel 축에 수직이어야 한다(축과 평행하면 그리퍼가 과실을 관통해야 한다).
        assert abs(float(np.dot(pose.approach_direction_unit, pose.pedicel_axis_unit))) < 1e-6

        # 접근 시작점은 과실 밖에 있어야 한다.
        assert np.linalg.norm(pose.pre_grasp_position_mm - pose.fruit_center_mm) > pose.fruit_radius_mm

        # 회전행렬은 정규직교이고 오른손 좌표계여야 한다(진짜 6-DOF).
        rotation = pose.rotation_matrix
        assert rotation.shape == (3, 3)
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        assert abs(float(np.linalg.det(rotation)) - 1.0) < 1e-6

    def test_unreachable_blade_offset_is_skipped(self) -> None:
        """고정 하드웨어 오프셋이 절단창과 맞지 않으면 스킵되어야 한다.

        blade_axial_offset_mm는 하드웨어 상수이므로 과실 크기에 따라 이상적 절단 거리와
        어긋날 수 있다. 어긋남이 허용치를 넘으면 칼날이 절단창을 벗어나 과실 어깨나
        모주(줄기)를 손상시킨다.
        """
        center = (0.0, 0.0, 500.0)
        radius = 11.0
        result = compute_grasp_cut_pose(
            make_sphere_points(center, radius),
            GraspCutPoseConfig(blade_axial_offset_mm=60.0, blade_offset_tolerance_mm=6.0),
            GeometricPedicelPoseEstimator(),
            fruit_id="F1",
            visible_ratio=0.9,
            pedicel_points_mm=make_pedicel_points(center, radius),
        )
        assert result.status is PoseStatus.SKIP_UNREACHABLE_OFFSET
        assert result.cut_offset_error_mm > 6.0

    def test_insufficient_points_is_skipped(self) -> None:
        result = compute_grasp_cut_pose(
            make_sphere_points((0.0, 0.0, 500.0), 11.0, count=10),
            GraspCutPoseConfig(min_points_for_sphere_fit=30),
            GeometricPedicelPoseEstimator(),
            fruit_id="F1",
            visible_ratio=0.9,
        )
        assert result.status is PoseStatus.SKIP_INSUFFICIENT_POINTS


# ---------------------------------------------------------------------------
# Deprojection
# ---------------------------------------------------------------------------

class TestDeprojection:
    def test_center_pixel_maps_to_optical_axis(self) -> None:
        intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=50.0, cy=50.0)
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[50, 50] = 500.0
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True

        points = deproject_depth_to_points(depth, mask, intrinsics)
        assert points.shape == (1, 3)
        assert np.allclose(points[0], [0.0, 0.0, 500.0], atol=1e-3)

    def test_zero_depth_pixels_are_dropped(self) -> None:
        intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=5.0, cy=5.0)
        depth = np.zeros((10, 10), dtype=np.float32)
        mask = np.ones((10, 10), dtype=bool)
        assert deproject_depth_to_points(depth, mask, intrinsics).shape[0] == 0

    def test_empty_mask_returns_empty(self) -> None:
        intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=5.0, cy=5.0)
        depth = np.full((10, 10), 500.0, dtype=np.float32)
        assert deproject_depth_to_points(depth, np.zeros((10, 10), bool), intrinsics).shape[0] == 0

    def test_invalid_intrinsics_rejected(self) -> None:
        with pytest.raises(ValueError):
            CameraIntrinsics(fx=0.0, fy=600.0, cx=5.0, cy=5.0)


# ---------------------------------------------------------------------------
# Pipeline 통합
# ---------------------------------------------------------------------------

def build_test_frame(
    truss_id: str = "T1", fruit_rgb: tuple[int, int, int] = (215, 30, 25)
) -> FrameInput:
    """과실 1개(+ Pedicel)를 담은 물리적으로 일관된 테스트 프레임.

    Args:
        truss_id: 트러스 식별자.
        fruit_rgb: 과실 색상. 기본값은 완숙(red) 색상이며, 미숙 단계 경로를 테스트할 때는
            브레이커 계열 색상(예: (226, 148, 38))을 넘긴다.
    """
    height, width = 240, 320
    intrinsics = CameraIntrinsics(fx=615.0, fy=615.0, cx=width / 2, cy=height / 2)

    rgb = np.full((height, width, 3), (40, 95, 35), dtype=np.uint8)
    depth = np.full((height, width), 900.0, dtype=np.float32)

    surface_depth, radius_mm = 520.0, 11.0
    center_x, center_y = width // 2, height // 2
    radius_px = int(round(radius_mm * intrinsics.fx / surface_depth))
    mm_per_px = surface_depth / intrinsics.fx

    cv2.circle(rgb, (center_x, center_y), radius_px, fruit_rgb, thickness=-1)
    y_grid, x_grid = np.ogrid[:height, :width]
    squared = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    inside = squared <= radius_px**2
    depth[inside] = (surface_depth - np.sqrt(np.maximum(radius_px**2 - squared, 0)) * mm_per_px)[
        inside
    ]

    # Pedicel
    pedicel_length_px = int(round(14.0 / mm_per_px))
    half_width_px = max(int(round(1.6 / mm_per_px)), 1)
    top_y = center_y - radius_px - pedicel_length_px
    bottom_y = center_y - radius_px
    cv2.rectangle(
        rgb,
        (center_x - half_width_px, top_y),
        (center_x + half_width_px, bottom_y),
        fruit_rgb,
        thickness=-1,
    )
    depth[top_y : bottom_y + 1, center_x - half_width_px : center_x + half_width_px + 1] = (
        surface_depth - 2.0
    )

    return FrameInput(
        rgb=rgb,
        depth_mm=depth,
        intrinsics=intrinsics,
        rgb_timestamp_ms=1000.0,
        depth_timestamp_ms=1002.0,
        truss_id=truss_id,
    )


def make_pipeline(config: PipelineConfig, **overrides) -> HarvestPreprocessingPipeline:
    components: dict[str, object] = {
        "segmentation_model": ClassicalColorSegmentationModel(
            min_instance_area_px=150, pedicel_removal_kernel_px=9
        ),
        "ripeness_classifier": RuleBasedRipenessClassifier(),
        "pedicel_estimator": GeometricPedicelPoseEstimator(),
    }
    components.update(overrides)
    return HarvestPreprocessingPipeline(config=config, **components)  # type: ignore[arg-type]


class TestPipelineIntegration:
    def test_end_to_end_produces_harvestable_fruit(self, config: PipelineConfig) -> None:
        result = make_pipeline(config).process_truss(build_test_frame())

        assert result.aborted_reason is None
        assert result.detected_fruit_count >= 1
        assert len(result.harvestable) >= 1

        harvest = result.harvestable[0]
        assert harvest.pose_result is not None
        assert harvest.pose_result.pose is not None
        assert 15.0 <= harvest.pose_result.pose.fruit_radius_mm * 2 <= 30.0

    def test_kpi_summary_is_consistent(self, config: PipelineConfig) -> None:
        result = make_pipeline(config).process_truss(build_test_frame())
        summary = result.kpi_summary()

        assert summary["evaluated_fruit_count"] == len(result.fruit_results)
        assert summary["ready_to_harvest_count"] == len(result.harvestable)
        counted = (
            int(summary["ready_to_harvest_count"])
            + int(summary["skipped_by_ripeness_gate"])
            + int(summary["skipped_by_pose_stage"])
            + int(summary["fruit_error_count"])
        )
        assert counted == summary["evaluated_fruit_count"]

    def test_sensor_desync_aborts_truss(self, config: PipelineConfig) -> None:
        frame = build_test_frame()
        desynced = FrameInput(
            rgb=frame.rgb,
            depth_mm=frame.depth_mm,
            intrinsics=frame.intrinsics,
            rgb_timestamp_ms=1000.0,
            depth_timestamp_ms=1500.0,  # 500ms 드리프트
            truss_id="T-DESYNC",
        )
        result = make_pipeline(config).process_truss(desynced)

        assert result.aborted_reason is not None
        assert "SensorSyncError" in result.aborted_reason
        assert result.fruit_results == ()

    def test_segmentation_failure_aborts_truss_gracefully(self, config: PipelineConfig) -> None:
        """모델 장애는 예외로 프로세스를 죽이지 않고 aborted_reason으로 보고되어야 한다."""

        class BrokenSegmentationModel:
            def predict(self, rgb):  # type: ignore[no-untyped-def]
                raise RuntimeError("CUDA out of memory")

        result = make_pipeline(
            config, segmentation_model=BrokenSegmentationModel()
        ).process_truss(build_test_frame())

        assert result.aborted_reason is not None
        assert "SegmentationModelError" in result.aborted_reason

    def test_single_fruit_error_does_not_abort_truss(self, config: PipelineConfig) -> None:
        """개별 과실 오류가 트러스 전체를 중단시키지 않아야 한다(핵심 안정성 요구사항)."""

        class ExplodingClassifier:
            def predict(self, patches_rgb):  # type: ignore[no-untyped-def]
                raise RuntimeError("모델 추론 중 예기치 않은 오류")

        result = make_pipeline(config, ripeness_classifier=ExplodingClassifier()).process_truss(
            build_test_frame()
        )

        assert result.aborted_reason is None, "트러스 처리는 계속되어야 한다"
        assert len(result.fruit_results) >= 1
        assert all(r.outcome is FruitOutcome.FRUIT_ERROR for r in result.fruit_results)
        assert result.fruit_results[0].error_message is not None

    def test_invalid_frame_shape_aborts(self, config: PipelineConfig) -> None:
        frame = FrameInput(
            rgb=np.zeros((100, 100, 3), dtype=np.uint8),
            depth_mm=np.zeros((50, 50), dtype=np.float32),  # shape 불일치
            intrinsics=CameraIntrinsics(fx=600.0, fy=600.0, cx=50.0, cy=50.0),
            rgb_timestamp_ms=0.0,
            depth_timestamp_ms=0.0,
            truss_id="T-BAD",
        )
        result = make_pipeline(config).process_truss(frame)
        assert result.aborted_reason is not None
        assert "InvalidImageError" in result.aborted_reason


class TestLogging:
    def test_writes_structured_json_lines(self, config: PipelineConfig) -> None:
        """JSON Lines 로그가 KPI 집계에 필요한 구조화 필드를 담아야 한다."""
        import json

        make_pipeline(config).process_truss(build_test_frame(truss_id="T-LOG"))

        log_path = config.logging.log_dir / config.logging.json_lines_filename
        assert log_path.is_file()

        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert records, "로그 레코드가 기록되어야 한다"
        for record in records:
            assert {"timestamp", "level", "logger", "message"} <= record.keys()

        stage_names = {
            record["extra"]["stage"] for record in records if "stage" in record.get("extra", {})
        }
        assert {"stage0_common", "segmentation"} <= stage_names

        completions = [record for record in records if record["message"] == "truss_completed"]
        assert completions
        assert completions[0]["extra"]["truss_id"] == "T-LOG"

    def test_stage_timer_records_elapsed_time(self, tmp_path: Path) -> None:
        import json

        from harvest_pipeline.logging_utils import get_logger, stage_timer

        logging_config = LoggingConfig(log_dir=tmp_path / "logs")
        logger = get_logger("test.stage_timer", logging_config)
        with stage_timer(logger, "unit_stage", fruit_id="F9") as ctx:
            ctx["custom_field"] = 42

        log_path = logging_config.log_dir / logging_config.json_lines_filename
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        end_records = [r for r in records if r["message"] == "stage_end"]
        assert end_records
        assert "elapsed_ms" in end_records[0]["extra"]
        assert end_records[0]["extra"]["custom_field"] == 42


# ---------------------------------------------------------------------------
# 회귀 방지 — 리뷰에서 실측 확인된 결함들이 다시 들어오지 않도록 고정한다.
# ---------------------------------------------------------------------------

class TestDepthRoundtripRegression:
    """`fill_depth_and_denoise`의 인페인팅 왕복 정확성(CRITICAL 회귀 방지)."""

    def test_valid_pixels_are_not_corrupted_by_inpainting(self) -> None:
        """hole을 채우는 과정이 유효 화소의 값을 변형하면 안 된다.

        과거 구현은 (a) cv2.normalize의 정변환(offset 0)과 역변환(offset D_min)이 어긋나
        400mm를 621mm로 만들었고(오차 +221mm — 목표 정밀도 ±2mm의 100배), (b) hole뿐 아니라
        전체 배열을 uint8 양자화 결과로 덮어써 과실 구면 곡률을 파괴했다.
        """
        depth = np.full((80, 80), 400.0, dtype=np.float32)
        depth[:, 40:] = 900.0
        depth[30:36, 30:36] = 0.0  # hole

        # bilateral filter가 값을 바꾸므로 영향이 최소화되는 설정을 쓴다.
        config = DepthFilterConfig(
            bilateral_diameter=1, bilateral_sigma_color=0.01, bilateral_sigma_space=0.01
        )
        result = s0.fill_depth_and_denoise(depth, config)

        # hole에서 멀리 떨어진 유효 화소는 원본 값을 유지해야 한다.
        assert abs(float(result[5, 5]) - 400.0) < 0.5
        assert abs(float(result[5, 70]) - 900.0) < 0.5
        # hole은 채워졌고, 값이 물리적으로 타당한 범위 안에 있어야 한다.
        assert np.count_nonzero(result == 0) == 0
        filled = result[30:36, 30:36]
        assert filled.min() >= 300.0 and filled.max() <= 1000.0

    def test_no_valid_pixels_does_not_crash(self) -> None:
        """유효 depth가 하나도 없는 프레임에서 크래시하면 트러스 전체가 죽는다."""
        depth = np.zeros((40, 40), dtype=np.float32)
        result = s0.fill_depth_and_denoise(depth, DepthFilterConfig())
        assert result.shape == depth.shape

    def test_uniform_valid_depth_does_not_crash(self) -> None:
        """모든 유효 depth가 동일하면 (D_max - D_min) = 0이 되어 0 나눗셈 위험이 있다."""
        depth = np.full((40, 40), 500.0, dtype=np.float32)
        depth[10:15, 10:15] = 0.0
        result = s0.fill_depth_and_denoise(depth, DepthFilterConfig())
        assert np.count_nonzero(result == 0) == 0
        assert abs(float(result[0, 0]) - 500.0) < 1.0


class TestCircularHueRegression:
    """Hue 순환(circular) 처리(CRITICAL 회귀 방지)."""

    def test_hue_near_360_is_classified_as_red(self) -> None:
        """Hue 350도(순수 적색)가 green으로 오분류되면 완숙과가 전량 스킵된다.

        과거 구현은 선형 거리로 최근접 기준값을 찾아, Hue 350도가 green(55도)까지의 거리
        295를 red(0도)까지의 거리 350보다 가깝다고 판단해 green으로 분류했다.
        """
        # OpenCV Hue 175 == 350도.
        hsv_patch = np.full((20, 1, 3), (175, 220, 200), dtype=np.uint8)
        rgb_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2RGB)

        prediction = RuleBasedRipenessClassifier().predict([rgb_patch])
        assert prediction.stage is RipenessStage.RED

    def test_hues_straddling_zero_average_to_red(self) -> None:
        """0도 양쪽에 걸친 Hue들(2도, 358도)의 원형 평균은 적색이어야 한다."""
        low = cv2.cvtColor(np.full((10, 1, 3), (1, 220, 200), dtype=np.uint8), cv2.COLOR_HSV2RGB)
        high = cv2.cvtColor(np.full((10, 1, 3), (179, 220, 200), dtype=np.uint8), cv2.COLOR_HSV2RGB)

        prediction = RuleBasedRipenessClassifier().predict([low, high])
        assert prediction.stage in (RipenessStage.RED, RipenessStage.LIGHT_RED)

    def test_confidence_is_consistent_across_patch_counts(self) -> None:
        """단색 과실의 신뢰도가 패치 수에 따라 흔들리면 임계값이 무의미해진다.

        과거 구현은 신뢰도가 (득표수/패치수) 이산값이라 5패치에서 {0.2,...,1.0}만 가능했고,
        임계값 0.90의 실효값이 1.0(만장일치)이 되었다. 게다가 패치가 4개로 줄면 4/4=1.0이
        통과해 패치가 적을 때 오히려 유리해지는 역전이 있었다.
        """
        rgb = np.full((160, 160, 3), (20, 20, 20), dtype=np.uint8)
        mask = np.zeros((160, 160), dtype=bool)
        cv2.circle(rgb, (80, 80), 40, (215, 30, 25), thickness=-1)
        cv2.circle(mask.view(np.uint8), (80, 80), 40, 1, thickness=-1)

        classifier = RuleBasedRipenessClassifier()
        confidences = []
        for count in (3, 5, 9):
            patches = sample_multi_patch_regions(
                rgb, mask, RipenessGateConfig(multi_patch_sample_count=count)
            )
            confidences.append(classifier.predict(patches).confidence)

        assert all(c > 0.85 for c in confidences), f"단색 과실 신뢰도가 낮음: {confidences}"
        assert max(confidences) - min(confidences) < 0.15, f"패치 수에 따라 흔들림: {confidences}"

    def test_left_right_uneven_coloring_lowers_confidence(self) -> None:
        """좌우로 발색이 다른 과실은 낮은 신뢰도를 받아야 한다(체크리스트 A-3).

        과거 구현은 y축으로만 등분해 좌우 패턴이 각 stripe 내부에서 평균되어 소멸했고,
        그 결과 절반이 미숙인 과실을 conf 1.0으로 "자신 있게" 오판정했다.
        """
        rgb = np.full((160, 160, 3), (20, 20, 20), dtype=np.uint8)
        mask = np.zeros((160, 160), dtype=bool)
        cv2.circle(mask.view(np.uint8), (80, 80), 40, 1, thickness=-1)
        rgb[:, :80][mask[:, :80]] = (215, 30, 25)  # 좌: 적색
        rgb[:, 80:][mask[:, 80:]] = (90, 170, 60)  # 우: 녹색

        patches = sample_multi_patch_regions(rgb, mask, RipenessGateConfig())
        prediction = RuleBasedRipenessClassifier().predict(patches)
        assert prediction.confidence < 0.9, f"좌우 불균일을 포착하지 못함: {prediction.confidence}"


class TestCrackingSizeInvarianceRegression:
    """열과 검출의 크기 무관성(MAJOR 회귀 방지)."""

    @pytest.mark.parametrize("diameter_px", [20, 40, 80])
    def test_smooth_fruit_is_not_flagged_regardless_of_size(self, diameter_px: int) -> None:
        """윤곽선을 엣지로 세면 작은 과실일수록 열과로 오판정된다.

        과거 구현은 마스크 전체에서 Canny 엣지를 세어 과실 테두리가 포함됐고, 밀도가 약
        2/반지름으로 감소해 지름 17px 미만에서 정상 과실도 임계값 0.12를 넘겼다.
        """
        size = diameter_px * 3
        radius = diameter_px // 2
        rgb = np.full((size, size, 3), (40, 95, 35), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=bool)
        center = size // 2
        cv2.circle(rgb, (center, center), radius, (215, 30, 25), thickness=-1)
        cv2.circle(mask.view(np.uint8), (center, center), radius, 1, thickness=-1)

        assert detect_cracking(rgb, mask, RipenessGateConfig()) is False

    def test_cracked_fruit_is_still_flagged(self) -> None:
        """크기 무관성을 얻는 과정에서 실제 균열 검출력을 잃지 않아야 한다."""
        rgb = np.full((200, 200, 3), (40, 95, 35), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=bool)
        cv2.circle(rgb, (100, 100), 45, (215, 30, 25), thickness=-1)
        cv2.circle(mask.view(np.uint8), (100, 100), 45, 1, thickness=-1)
        for offset in range(-30, 31, 6):
            cv2.line(rgb, (100 + offset, 65), (100 + offset - 8, 135), (55, 8, 8), thickness=2)

        assert detect_cracking(rgb, mask, RipenessGateConfig()) is True


class TestVisibleRatioRegression:
    """가시 비율 추정(CRITICAL 회귀 방지)."""

    @staticmethod
    def _instance(mask: np.ndarray) -> object:
        from harvest_pipeline.interfaces import SegmentationInstance

        ys, xs = np.where(mask)
        return SegmentationInstance(
            instance_id=1,
            class_label="fruit",
            mask=mask,
            bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            confidence=1.0,
        )

    def test_occluded_fruit_scores_lower_than_intact(self) -> None:
        """절반 가려진 과실이 온전한 과실보다 낮은 가시 비율을 받아야 한다.

        과거 구현(마스크면적/bbox면적)은 bbox가 그 마스크에서 도출되므로 가림으로 마스크가
        줄면 bbox도 함께 줄어 비율이 보존됐다. 실측 결과 우측 절반이 가려진 과실이 0.993,
        온전한 과실이 0.965로 역전되어 occlusion 게이트가 전혀 발동하지 않았다.
        """
        from harvest_pipeline.pipeline import estimate_visible_ratio

        intact = np.zeros((160, 160), dtype=bool)
        cv2.circle(intact.view(np.uint8), (80, 80), 40, 1, thickness=-1)

        half_occluded = intact.copy()
        half_occluded[:, 80:] = False  # 우측 절반 가림

        intact_ratio = estimate_visible_ratio(self._instance(intact))
        occluded_ratio = estimate_visible_ratio(self._instance(half_occluded))

        assert intact_ratio > 0.9, f"온전한 과실 비율이 낮음: {intact_ratio}"
        assert occluded_ratio < intact_ratio - 0.2, (
            f"가림을 감지하지 못함: intact={intact_ratio}, occluded={occluded_ratio}"
        )

    def test_empty_mask_returns_zero(self) -> None:
        from harvest_pipeline.pipeline import estimate_visible_ratio

        from harvest_pipeline.interfaces import SegmentationInstance

        empty = SegmentationInstance(
            instance_id=1,
            class_label="fruit",
            mask=np.zeros((50, 50), dtype=bool),
            bbox_xyxy=(0, 0, 1, 1),
            confidence=1.0,
        )
        assert estimate_visible_ratio(empty) == 0.0


class TestVoxelDownsampleAccuracy:
    """다운샘플링이 실제로 평균을 계산하는지(MAJOR 회귀 방지)."""

    def test_points_in_same_voxel_are_averaged(self) -> None:
        """개수 감소만 검증하면 평균 계산을 합계로 망가뜨려도 통과한다."""
        # voxel 크기 10mm — 아래 세 점은 모두 같은 격자에 속한다.
        points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32)
        result = s0.voxel_downsample(points, PointCloudConfig(voxel_size_mm=10.0))

        assert result.shape == (1, 3)
        assert np.allclose(result[0], [2.0, 2.0, 2.0], atol=1e-5)

    def test_points_in_different_voxels_are_kept_separate(self) -> None:
        points = np.array([[1.0, 1.0, 1.0], [100.0, 100.0, 100.0]], dtype=np.float32)
        result = s0.voxel_downsample(points, PointCloudConfig(voxel_size_mm=10.0))
        assert result.shape == (2, 3)


class TestPipelineRobustnessRegression:
    """오케스트레이션 견고성(MAJOR 회귀 방지)."""

    def test_fatal_exception_is_reraised_not_swallowed(self, config: PipelineConfig) -> None:
        """MemoryError를 과실 단위 오류로 삼키면 진짜 원인이 로그에서 은폐된다.

        과거 구현은 `except Exception`으로 MemoryError까지 잡아 30개 과실을 모두
        FRUIT_ERROR로 기록하며 루프를 계속 돌렸다.
        """

        class MemoryExhaustingClassifier:
            def predict(self, patches_rgb):  # type: ignore[no-untyped-def]
                raise MemoryError("heap exhausted")

        pipeline = make_pipeline(config, ripeness_classifier=MemoryExhaustingClassifier())
        with pytest.raises(MemoryError):
            pipeline.process_truss(build_test_frame())

    def test_non_domain_exception_is_recorded_as_aborted(self, config: PipelineConfig) -> None:
        """도메인 예외가 아닌 오류(cv2.error 등)도 aborted_reason으로 기록되어야 한다.

        과거 구현은 `except HarvestPipelineError`로 좁게 잡아, OpenCV 내부 오류가 그대로
        뚫고 나가 "프레임 오류는 aborted_reason에 기록한다"는 계약이 깨졌다.
        """

        class CvErrorSegmentationModel:
            def predict(self, rgb):  # type: ignore[no-untyped-def]
                raise cv2.error("OpenCV internal failure")

        result = make_pipeline(
            config, segmentation_model=CvErrorSegmentationModel()
        ).process_truss(build_test_frame())

        assert result.aborted_reason is not None
        assert result.fruit_results == ()

    def test_vision_inference_timeout_is_enforced(self, tmp_path: Path) -> None:
        """비전 추론이 제한 시간을 넘기면 호출자가 제어권을 되찾아야 한다."""
        import time as _time

        from harvest_pipeline.config import TimeoutConfig

        class SlowSegmentationModel:
            def predict(self, rgb):  # type: ignore[no-untyped-def]
                _time.sleep(1.0)
                return SegmentationResult()

        slow_config = PipelineConfig(
            logging=LoggingConfig(log_dir=tmp_path / "logs"),
            timeouts=TimeoutConfig(vision_inference_timeout_s=0.05),
        )
        with make_pipeline(slow_config, segmentation_model=SlowSegmentationModel()) as pipeline:
            started = _time.perf_counter()
            result = pipeline.process_truss(build_test_frame())
            elapsed = _time.perf_counter() - started

        assert result.aborted_reason is not None
        assert "HardwareTimeoutError" in result.aborted_reason
        assert elapsed < 0.9, f"타임아웃이 호출자를 해제하지 못했다: {elapsed:.2f}s"

    def test_truss_timeout_yields_partial_result(self, tmp_path: Path) -> None:
        """트러스 처리 시간이 초과되면 부분 결과와 timed_out 플래그를 반환해야 한다."""
        from harvest_pipeline.config import TimeoutConfig

        tight_config = PipelineConfig(
            logging=LoggingConfig(log_dir=tmp_path / "logs"),
            timeouts=TimeoutConfig(truss_processing_timeout_s=1e-6),
        )
        result = make_pipeline(tight_config).process_truss(build_test_frame())

        assert result.timed_out is True
        assert len(result.fruit_results) < result.detected_fruit_count
        assert int(result.kpi_summary()["unprocessed_count"]) > 0

    def test_static_structure_mask_is_applied(self, config: PipelineConfig) -> None:
        """구조물 마스크 경로를 실행하는 테스트가 없어, 호출을 지워도 전부 통과했다."""
        frame = build_test_frame()
        structure_mask = np.zeros(frame.rgb.shape[:2], dtype=bool)
        structure_mask[0:10, :] = True

        masked_frame = FrameInput(
            rgb=frame.rgb,
            depth_mm=frame.depth_mm,
            intrinsics=frame.intrinsics,
            rgb_timestamp_ms=frame.rgb_timestamp_ms,
            depth_timestamp_ms=frame.depth_timestamp_ms,
            truss_id="T-MASK",
            static_structure_mask=structure_mask,
        )
        rgb, depth = make_pipeline(config).run_stage0(masked_frame)

        assert int(rgb[0:10, :].sum()) == 0
        assert float(depth[0:10, :].sum()) == 0.0
        assert int(rgb[20:, :].sum()) > 0  # 마스크 밖은 보존

    def test_white_reference_is_forwarded_to_calibration(self, config: PipelineConfig) -> None:
        """FrameInput.white_reference_rgb가 실제로 캘리브레이션에 전달되어야 한다."""
        frame = build_test_frame()
        with_reference = FrameInput(
            rgb=frame.rgb,
            depth_mm=frame.depth_mm,
            intrinsics=frame.intrinsics,
            rgb_timestamp_ms=frame.rgb_timestamp_ms,
            depth_timestamp_ms=frame.depth_timestamp_ms,
            truss_id="T-WB",
            white_reference_rgb=(90.0, 140.0, 200.0),  # 청색 편향 광원
        )

        baseline_rgb, _ = make_pipeline(config).run_stage0(frame)
        adjusted_rgb, _ = make_pipeline(config).run_stage0(with_reference)

        assert not np.array_equal(baseline_rgb, adjusted_rgb), "화이트 레퍼런스가 무시됨"

    def test_camera_to_base_transform_shifts_pose(self, config: PipelineConfig) -> None:
        """변환행렬을 주면 산출 포즈가 로봇 베이스 좌표계로 이동해야 한다."""
        frame = build_test_frame()
        translation = np.array([100.0, 200.0, -300.0])
        transform = np.eye(4)
        transform[:3, 3] = translation

        transformed_frame = FrameInput(
            rgb=frame.rgb,
            depth_mm=frame.depth_mm,
            intrinsics=frame.intrinsics,
            rgb_timestamp_ms=frame.rgb_timestamp_ms,
            depth_timestamp_ms=frame.depth_timestamp_ms,
            truss_id="T-TF",
            camera_to_base_transform=transform,
        )

        baseline = make_pipeline(config).process_truss(frame)
        shifted = make_pipeline(config).process_truss(transformed_frame)

        assert baseline.harvestable and shifted.harvestable
        base_center = baseline.harvestable[0].pose_result.pose.fruit_center_mm  # type: ignore[union-attr]
        shifted_center = shifted.harvestable[0].pose_result.pose.fruit_center_mm  # type: ignore[union-attr]

        assert np.allclose(shifted_center - base_center, translation, atol=1.0)

    def test_all_fruit_outcomes_are_reachable(self, config: PipelineConfig) -> None:
        """FruitOutcome 네 값이 모두 실제 파이프라인 경로에서 산출되어야 한다."""
        observed: set[FruitOutcome] = set()

        # READY_TO_HARVEST
        observed.update(r.outcome for r in make_pipeline(config).process_truss(build_test_frame()).fruit_results)

        # SKIPPED_BY_RIPENESS_GATE — 미숙(breaker) 색상 과실
        immature = build_test_frame(truss_id="T-IMMATURE", fruit_rgb=(226, 148, 38))
        observed.update(r.outcome for r in make_pipeline(config).process_truss(immature).fruit_results)

        # SKIPPED_BY_POSE_STAGE — Pedicel 축 신뢰도를 통과 불가로 설정
        from harvest_pipeline.config import GraspCutPoseConfig as _GCP

        strict = PipelineConfig(
            logging=config.logging,
            grasp_cut_pose=_GCP(min_axis_confidence=0.999),
        )
        observed.update(
            r.outcome for r in make_pipeline(strict).process_truss(build_test_frame()).fruit_results
        )

        # FRUIT_ERROR
        class BrokenClassifier:
            def predict(self, patches_rgb):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")

        observed.update(
            r.outcome
            for r in make_pipeline(config, ripeness_classifier=BrokenClassifier())
            .process_truss(build_test_frame())
            .fruit_results
        )

        assert observed == set(FruitOutcome), f"미커버 outcome: {set(FruitOutcome) - observed}"


class TestLoggingRobustnessRegression:
    def test_reserved_extra_keys_do_not_crash(self, tmp_path: Path) -> None:
        """LogRecord 예약어를 extra로 넘기면 logging이 KeyError를 던져 파이프라인이 죽는다.

        stage_timer의 ctx는 호출자가 임의 키를 넣을 수 있으므로 사전 방어가 필요하다.
        """
        from harvest_pipeline.logging_utils import get_logger, stage_timer

        logging_config = LoggingConfig(log_dir=tmp_path / "logs")
        logger = get_logger("test.reserved_keys", logging_config)

        with stage_timer(logger, "unit_stage") as ctx:
            ctx["module"] = "should_not_crash"
            ctx["process"] = 1234
            ctx["message"] = "shadowed"

        log_path = logging_config.log_dir / logging_config.json_lines_filename
        assert log_path.is_file()
