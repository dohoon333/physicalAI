#!/usr/bin/env python3
"""전처리 파이프라인 end-to-end 데모.

실제 RGB-D 카메라 없이도 파이프라인 전체 흐름을 검증할 수 있도록, 온실 트러스를 모사한
합성 프레임(과실 여러 개 + 초록 잎 배경 + Pedicel 형상)을 생성해 실행한다.

합성 기하는 카메라 내부 파라미터(intrinsics)와 **물리적으로 일관되게** 구성한다. 즉 원하는
과실 실물 반지름(mm)과 촬영 거리로부터 픽셀 반지름을 역산하고, Depth 기복도 동일한 변환
계수(mm/px = depth / fx)를 사용한다. 이 일관성이 없으면 Stage B의 구 피팅 결과가 실제와
다른 크기로 나와 허용 범위 검사에서 걸러진다.

실행:
    python examples/run_pipeline_demo.py
    python examples/run_pipeline_demo.py --config configs/default_pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harvest_pipeline.config import PipelineConfig  # noqa: E402
from harvest_pipeline.interfaces import (  # noqa: E402
    ClassicalColorSegmentationModel,
    GeometricPedicelPoseEstimator,
    RuleBasedRipenessClassifier,
)
from harvest_pipeline.pipeline import (  # noqa: E402
    CameraIntrinsics,
    FrameInput,
    HarvestPreprocessingPipeline,
)

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
BACKGROUND_DEPTH_MM = 900.0
LEAF_BACKGROUND_RGB = (40, 95, 35)

# 근접 정밀 촬영 구성(PRD 4장 D405급 센서, 체크리스트 D-3 "근거리" 대응).
DEMO_INTRINSICS = CameraIntrinsics(fx=615.0, fy=615.0, cx=IMAGE_WIDTH / 2, cy=IMAGE_HEIGHT / 2)

PEDICEL_LENGTH_MM = 14.0
PEDICEL_HALF_WIDTH_MM = 1.6


@dataclass(frozen=True, slots=True)
class SyntheticFruit:
    """합성 과실 하나의 물리적 명세."""

    center_xy: tuple[int, int]
    color_rgb: tuple[int, int, int]
    surface_depth_mm: float
    radius_mm: float
    label: str

    def radius_px(self, fx: float) -> int:
        """물리 반지름(mm)을 촬영 거리에서의 픽셀 반지름으로 환산한다."""
        return max(int(round(self.radius_mm * fx / self.surface_depth_mm)), 1)

    def mm_per_px(self, fx: float) -> float:
        """이 과실 촬영 거리에서의 픽셀→mm 변환 계수."""
        return self.surface_depth_mm / fx


def build_synthetic_frame() -> tuple[np.ndarray, np.ndarray, list[SyntheticFruit]]:
    """트러스를 모사한 합성 RGB/Depth 프레임을 만든다.

    - 잘 익은 과실 2개(수확 대상), 아직 덜 익은 과실 1개(스킵 대상)를 배치.
    - Depth는 과실이 배경보다 카메라에 가깝도록(값이 작도록) 구성한다.

    덜 익은 과실을 완전한 초록(Green 단계)이 아니라 색이 돌기 시작한 단계로 만든 이유:
    데모가 사용하는 규칙 기반 색상 세그멘테이션은 잎과 색이 겹치는 Green 단계 과실을
    원리적으로 탐지할 수 없다(ClassicalColorSegmentationModel docstring 참고). 색이 돌기
    시작한 단계는 탐지되면서도 수확 대상(라이트레드~레드)에는 해당하지 않으므로, 게이트가
    미숙과를 걸러내는 동작을 보여줄 수 있다.
    """
    rgb = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), LEAF_BACKGROUND_RGB, dtype=np.uint8)
    depth = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND_DEPTH_MM, dtype=np.float32)

    # 색상은 RuleBasedRipenessClassifier의 단계별 Hue 기준값에 맞춰 선택했다.
    # (red≈0°, light_red≈5°, pink≈10°, turning≈25°, breaker≈40°, green≈55°)
    fruits = [
        SyntheticFruit((170, 240), (215, 30, 25), 520.0, 11.0, "red(수확대상)"),
        SyntheticFruit((340, 260), (235, 95, 80), 540.0, 12.0, "light_red(수확대상)"),
        SyntheticFruit((500, 230), (226, 148, 38), 530.0, 11.5, "breaker(미숙-스킵)"),
    ]

    for fruit in fruits:
        _draw_fruit(rgb, depth, fruit)
        _draw_pedicel(rgb, depth, fruit)

    return rgb, depth, fruits


def _draw_fruit(rgb: np.ndarray, depth: np.ndarray, fruit: SyntheticFruit) -> None:
    """과실 본체를 RGB에 원으로, Depth에 구면 기복으로 기록한다.

    Depth가 구면을 따라 변해야 Stage B의 구 피팅이 의미 있게 동작한다. 중심에 가까울수록
    카메라에 가깝고(값이 작고), 테두리로 갈수록 표면 깊이에 수렴한다.
    """
    cx, cy = fruit.center_xy
    radius_px = fruit.radius_px(DEMO_INTRINSICS.fx)
    mm_per_px = fruit.mm_per_px(DEMO_INTRINSICS.fx)

    cv2.circle(rgb, (cx, cy), radius_px, fruit.color_rgb, thickness=-1)

    y_grid, x_grid = np.ogrid[: depth.shape[0], : depth.shape[1]]
    squared_distance_px = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    inside = squared_distance_px <= radius_px**2

    # 구면 융기 높이: sqrt(r² - d²)를 픽셀 단위로 계산한 뒤 동일 계수로 mm 환산.
    bulge_mm = np.sqrt(np.maximum(radius_px**2 - squared_distance_px, 0)) * mm_per_px
    depth[inside] = (fruit.surface_depth_mm - bulge_mm)[inside]


def _draw_pedicel(rgb: np.ndarray, depth: np.ndarray, fruit: SyntheticFruit) -> None:
    """과실 상단에 Pedicel(꼭지)을 그린다.

    Stage B의 축 추정은 과실 상단부의 막대형 포인트 분포(linearity 지표)에 의존하므로,
    Pedicel 형상이 없으면 축 신뢰도가 낮아 스킵된다. 실제 시스템에서도 세그멘테이션이
    과실 본체 마스크와 Pedicel 마스크를 함께 제공해야 한다
    (interfaces.SegmentationInstance.combined_mask 참고).

    주의: 이미지 y축은 아래로 증가하지만 로봇/카메라 좌표계의 Y는 위를 향하도록 역투영되므로,
    Pedicel을 이미지상 위쪽(y가 작은 방향)에 그리면 3D에서는 -Y 방향에 놓인다. Stage B의
    축 부호 보정 로직이 이를 흡수하기 때문에 문제가 되지 않는다.
    """
    cx, cy = fruit.center_xy
    radius_px = fruit.radius_px(DEMO_INTRINSICS.fx)
    mm_per_px = fruit.mm_per_px(DEMO_INTRINSICS.fx)

    length_px = max(int(round(PEDICEL_LENGTH_MM / mm_per_px)), 2)
    half_width_px = max(int(round(PEDICEL_HALF_WIDTH_MM / mm_per_px)), 1)

    top_y = max(cy - radius_px - length_px, 0)
    bottom_y = max(cy - radius_px, 0)

    cv2.rectangle(
        rgb,
        (cx - half_width_px, top_y),
        (cx + half_width_px, bottom_y),
        fruit.color_rgb,
        thickness=-1,
    )
    # Pedicel은 과실 표면보다 약간 뒤쪽(카메라에서 멀게)에 위치한다고 가정.
    depth[top_y : bottom_y + 1, cx - half_width_px : cx + half_width_px + 1] = (
        fruit.surface_depth_mm - 2.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="방울토마토 전처리 파이프라인 데모")
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML 설정 파일 경로(생략 시 코드 기본값 사용)"
    )
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()

    pipeline = HarvestPreprocessingPipeline(
        config=config,
        segmentation_model=ClassicalColorSegmentationModel(
            min_instance_area_px=200, pedicel_removal_kernel_px=9
        ),
        ripeness_classifier=RuleBasedRipenessClassifier(),
        pedicel_estimator=GeometricPedicelPoseEstimator(),
        logger_name="harvest_pipeline.demo",
    )

    rgb, depth, fruits = build_synthetic_frame()
    print(f"\n합성 프레임 생성: 과실 {len(fruits)}개 ({', '.join(f.label for f in fruits)})")
    for fruit in fruits:
        print(
            f"  - {fruit.label:18s} 반지름 {fruit.radius_mm:.1f}mm "
            f"(= {fruit.radius_px(DEMO_INTRINSICS.fx)}px @ {fruit.surface_depth_mm:.0f}mm)"
        )
    print()

    frame = FrameInput(
        rgb=rgb,
        depth_mm=depth,
        intrinsics=DEMO_INTRINSICS,
        rgb_timestamp_ms=1000.0,
        depth_timestamp_ms=1003.0,  # 3ms 드리프트 — 허용치(15ms) 이내
        truss_id="DEMO-TRUSS-01",
    )

    result = pipeline.process_truss(frame)

    print("=" * 70)
    print("과실별 처리 결과")
    print("=" * 70)
    for fruit_result in result.fruit_results:
        print(json.dumps(fruit_result.as_log_fields(), ensure_ascii=False, indent=2))
        print("-" * 70)

    print("\n" + "=" * 70)
    print("트러스 KPI 요약")
    print("=" * 70)
    print(json.dumps(result.kpi_summary(), ensure_ascii=False, indent=2))

    log_path = config.logging.log_dir / config.logging.json_lines_filename
    print(f"\n구조화 로그(JSON Lines) 저장 위치: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
