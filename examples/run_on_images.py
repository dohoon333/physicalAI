#!/usr/bin/env python3
"""실제 RGB + Depth 파일로 전처리 파이프라인을 실행한다.

합성 데모(run_pipeline_demo.py)와 달리 디스크에 저장된 실제 촬영 데이터를 입력으로 받는다.

지원 입력 형식:
  RGB   : .png / .jpg / .bmp (8bit 3채널)
  Depth : .png (16bit, 값 단위 = mm)  또는  .npy (float/int, 값 단위 = mm)

RealSense로 데이터를 저장할 때 Depth를 16bit PNG(mm 단위)로 쓰는 것이 일반적이며,
`pyrealsense2`의 depth_scale이 0.001(=1mm)인 경우 raw 값이 곧 mm다.

사용 예:
    # 단일 프레임
    python examples/run_on_images.py --rgb data/truss01.png --depth data/truss01_depth.png \
        --fx 615 --fy 615 --cx 320 --cy 240

    # 폴더 일괄 처리(파일명 규칙: <name>.png + <name>_depth.png)
    python examples/run_on_images.py --input-dir data/ --fx 615 --fy 615 --cx 320 --cy 240

    # 현장별 설정 적용 + 결과 JSON 저장
    python examples/run_on_images.py --input-dir data/ --config configs/greenhouse_A.yaml \
        --fx 615 --fy 615 --cx 320 --cy 240 --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
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

RGB_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def load_rgb(path: Path) -> np.ndarray:
    """RGB 이미지를 (H, W, 3) uint8 RGB 순서로 읽는다(OpenCV는 BGR로 읽으므로 변환)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"RGB 이미지를 읽을 수 없습니다: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_depth_mm(path: Path) -> np.ndarray:
    """Depth를 (H, W) float32 배열(단위 mm)로 읽는다.

    .npy는 그대로, 16bit PNG는 raw 값을 mm로 해석한다. Depth 스케일이 1mm가 아닌 센서를
    쓴다면 --depth-scale 옵션으로 보정한다.
    """
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth 이미지를 읽을 수 없습니다: {path}")
    if depth.ndim != 2:
        raise ValueError(
            f"Depth는 단일 채널이어야 합니다(현재 shape={depth.shape}). "
            "3채널로 저장된 컬러맵 이미지는 사용할 수 없습니다 — raw 16bit로 다시 저장하세요."
        )
    return depth.astype(np.float32)


def find_frame_pairs(input_dir: Path, depth_suffix: str) -> list[tuple[Path, Path]]:
    """폴더에서 (RGB, Depth) 파일 쌍을 찾는다.

    규칙: `<name>.png`와 `<name><depth_suffix>.png`(또는 .npy)를 한 쌍으로 본다.
    """
    pairs: list[tuple[Path, Path]] = []
    for rgb_path in sorted(input_dir.iterdir()):
        if rgb_path.suffix.lower() not in RGB_SUFFIXES:
            continue
        if rgb_path.stem.endswith(depth_suffix):
            continue  # depth 파일 자체는 건너뛴다

        for candidate in (
            rgb_path.with_name(f"{rgb_path.stem}{depth_suffix}.png"),
            rgb_path.with_name(f"{rgb_path.stem}{depth_suffix}.npy"),
        ):
            if candidate.is_file():
                pairs.append((rgb_path, candidate))
                break
    return pairs


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="실제 RGB-D 파일로 방울토마토 전처리 파이프라인 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rgb", type=Path, help="단일 RGB 이미지 경로")
    source.add_argument("--input-dir", type=Path, help="RGB/Depth 쌍이 있는 폴더")

    parser.add_argument("--depth", type=Path, help="단일 Depth 파일 경로(--rgb와 함께 사용)")
    parser.add_argument(
        "--depth-suffix", default="_depth", help="폴더 처리 시 Depth 파일 접미사(기본: _depth)"
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Depth raw 값 → mm 변환 계수(기본 1.0 = 이미 mm 단위)",
    )

    # 카메라 내부 파라미터는 센서마다 다르므로 필수로 받는다. RealSense는
    # `pyrealsense2`의 get_intrinsics()로, 그 외에는 체커보드 캘리브레이션으로 구한다.
    parser.add_argument("--fx", type=float, required=True, help="초점거리 fx (px)")
    parser.add_argument("--fy", type=float, required=True, help="초점거리 fy (px)")
    parser.add_argument("--cx", type=float, required=True, help="주점 cx (px)")
    parser.add_argument("--cy", type=float, required=True, help="주점 cy (px)")

    parser.add_argument("--config", type=Path, help="YAML 설정(생략 시 코드 기본값)")
    parser.add_argument("--output", type=Path, help="결과를 JSON으로 저장할 경로")
    parser.add_argument(
        "--white-reference",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="ColorChecker 흰색 패치 측정 RGB값(생략 시 채널별 화이트밸런스 미적용)",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    if args.rgb is not None and args.depth is None:
        print("[오류] --rgb를 쓸 때는 --depth도 함께 지정해야 합니다.", file=sys.stderr)
        return 2

    config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    intrinsics = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
    white_reference = tuple(args.white_reference) if args.white_reference else None

    if args.rgb is not None:
        frame_pairs = [(args.rgb, args.depth)]
    else:
        if not args.input_dir.is_dir():
            print(f"[오류] 폴더를 찾을 수 없습니다: {args.input_dir}", file=sys.stderr)
            return 2
        frame_pairs = find_frame_pairs(args.input_dir, args.depth_suffix)
        if not frame_pairs:
            print(
                f"[오류] {args.input_dir}에서 RGB/Depth 쌍을 찾지 못했습니다. "
                f"파일명 규칙: <name>.png + <name>{args.depth_suffix}.png",
                file=sys.stderr,
            )
            return 2

    print(f"처리할 프레임: {len(frame_pairs)}개\n")

    # 학습된 모델이 준비되면 아래 세 컴포넌트만 교체하면 된다(파이프라인 코드 수정 불필요).
    pipeline = HarvestPreprocessingPipeline(
        config=config,
        segmentation_model=ClassicalColorSegmentationModel(),
        ripeness_classifier=RuleBasedRipenessClassifier(),
        pedicel_estimator=GeometricPedicelPoseEstimator(),
        logger_name="harvest_pipeline.images",
    )

    all_summaries: list[dict[str, object]] = []
    try:
        for rgb_path, depth_path in frame_pairs:
            try:
                rgb = load_rgb(rgb_path)
                depth_mm = load_depth_mm(depth_path) * args.depth_scale
            except (FileNotFoundError, ValueError) as exc:
                print(f"[건너뜀] {rgb_path.name}: {exc}", file=sys.stderr)
                continue

            if depth_mm.shape != rgb.shape[:2]:
                print(
                    f"[건너뜀] {rgb_path.name}: RGB {rgb.shape[:2]}와 "
                    f"Depth {depth_mm.shape} 해상도가 다릅니다(정렬된 프레임이 필요).",
                    file=sys.stderr,
                )
                continue

            frame = FrameInput(
                rgb=rgb,
                depth_mm=depth_mm,
                intrinsics=intrinsics,
                # 파일로 저장된 데이터는 이미 정렬된 프레임이라고 보고 동일 타임스탬프를 준다.
                # 실시간 스트림에서는 각 프레임의 실제 타임스탬프를 넣어야 동기화 검사가 의미를 갖는다.
                rgb_timestamp_ms=0.0,
                depth_timestamp_ms=0.0,
                truss_id=rgb_path.stem,
                white_reference_rgb=white_reference,
            )

            result = pipeline.process_truss(frame)
            summary = result.kpi_summary()
            all_summaries.append(summary)

            status = result.aborted_reason or "OK"
            print(
                f"  {rgb_path.name:30s} 탐지 {summary['detected_fruit_count']:2d}개 "
                f"→ 수확대상 {summary['ready_to_harvest_count']:2d}개 "
                f"({summary['elapsed_ms']:.0f}ms) {status}"
            )
    finally:
        pipeline.close()

    if not all_summaries:
        print("\n처리된 프레임이 없습니다.", file=sys.stderr)
        return 1

    total_detected = sum(int(s["detected_fruit_count"]) for s in all_summaries)
    total_ready = sum(int(s["ready_to_harvest_count"]) for s in all_summaries)
    print(f"\n합계: 탐지 {total_detected}개 → 수확대상 {total_ready}개")
    print(f"구조화 로그: {config.logging.log_dir / config.logging.json_lines_filename}")

    if args.output:
        args.output.write_text(
            json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"결과 저장: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
