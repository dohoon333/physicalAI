"""파이프라인 예외 계층.

설계 원칙: "정상적으로 예상되는 분기"(미숙과라서 스킵, 가림이 심해 스킵)는 예외로 표현하지
않는다 — 이는 result 객체(stage_a_ripeness.GateResult, stage_b_grasp_cut.PoseResult 등)의
status 필드로 나타낸다. 예외는 오직 "설계상 발생해서는 안 되는 진짜 오류"에만 사용한다.
이렇게 구분하지 않으면 정상 업무 흐름 제어에 예외를 남용하게 되어 호출부 코드가 지저분해지고,
실제 장애와 정상 스킵을 로그에서 구분하기도 어려워진다.
"""

from __future__ import annotations


class HarvestPipelineError(Exception):
    """모든 파이프라인 예외의 공통 베이스."""


class ConfigValidationError(HarvestPipelineError):
    """설정값이 유효하지 않을 때(스키마 검증 실패 등)."""


class SensorSyncError(HarvestPipelineError):
    """센서 간 시간/좌표 동기화 실패."""


class InvalidImageError(HarvestPipelineError):
    """입력 이미지의 shape/dtype/채널 수가 기대값과 다를 때."""


class InvalidPointCloudError(HarvestPipelineError):
    """포인트클라우드가 비어있거나 형식이 잘못되었을 때."""


class CalibrationError(HarvestPipelineError):
    """카메라-로봇 좌표계 캘리브레이션 실패(4장 Hand-Eye Calibration)."""


class SegmentationModelError(HarvestPipelineError):
    """인스턴스 세그멘테이션 모델 추론 실패(모델 로드 실패, 추론 타임아웃 등)."""


class RetryExhausted(HarvestPipelineError):
    """재시도 한도를 초과했지만 이는 상위 호출부가 정상적으로 스킵 처리해야 하는 상황이다.

    주의: 이 예외는 즉시 프로세스를 중단시키기 위한 것이 아니라, 재시도 루프를 벗어났음을
    명시적으로 알리기 위한 신호(signal)다. 호출부(pipeline.py)는 반드시 이를 잡아서
    스킵 상태의 result 객체로 변환해야 하며, 트러스 전체 처리를 중단해서는 안 된다.
    """


class HardwareTimeoutError(HarvestPipelineError):
    """비전 추론/파지·절단 실행이 설정된 타임아웃(TimeoutConfig)을 초과했을 때."""
