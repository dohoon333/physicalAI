"""구조화(JSON) 로깅 설정.

PRD의 대시보드(수확량/손상률/숙성 지도/미탐지율 시각화, 5.1-8항)는 결국 이 로그를 원본
데이터로 삼아 집계된다. 따라서 로그는 사람이 읽기 좋은 콘솔 출력과, 기계가 파싱하기 좋은
JSON Lines 파일 출력을 함께 남기도록 이중화한다.

- 콘솔: 개발/운영 중 즉시 확인용, 사람이 읽기 쉬운 포맷.
- 파일(JSON Lines, 로테이션): harvest_mission_id/fruit_id/stage/elapsed_ms/status 등
  구조화 필드를 담아, 이후 KPI 집계 배치job이 그대로 읽어 처리할 수 있게 한다.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Any

from harvest_pipeline.config import LoggingConfig

_CONFIGURED_LOGGERS: dict[str, LoggingConfig] = {}

# 로거 구성은 "확인 후 변경(check-then-act)"이므로 원자적이지 않다. 락 없이 두 스레드가
# 동시에 진입하면 (a) 핸들러가 중복 등록되어 로그가 두 배로 쌓이거나, (b) 한 스레드가
# close()한 핸들러를 다른 스레드가 emit 중에 사용해 "I/O operation on closed file"로
# 실패한다. 로거 구성은 드물게 일어나므로 락 경합 비용은 무시할 수 있다.
_CONFIG_LOCK = threading.Lock()

# `extra`로 들어온 키가 LogRecord의 예약 속성과 겹치면 logging 모듈이 KeyError를 던져
# 파이프라인이 죽는다. 호출자가 임의 필드를 넘길 수 있으므로(stage_timer의 ctx) 사전에
# 감지해 접두어를 붙여 회피한다.
_RESERVED_EXTRA_KEYS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
) | {"message", "asctime"}


def _sanitize_extra(fields: dict[str, Any]) -> dict[str, Any]:
    """LogRecord 예약어와 충돌하는 키에 접두어를 붙여 KeyError를 예방한다."""
    sanitized: dict[str, Any] = {}
    for key, value in fields.items():
        sanitized[f"x_{key}" if key in _RESERVED_EXTRA_KEYS else key] = value
    return sanitized

# logging.LogRecord가 기본으로 갖는 속성 목록(이 목록에 없는 속성만 "extra"로 간주해 JSON에 포함).
# "message"/"asctime"은 LogRecord 생성 시점에는 없지만, 다른 핸들러의 표준 Formatter가
# 먼저 실행되며 동일 record 객체에 부가적으로 세팅하는 필드라서(핸들러는 같은 record 인스턴스를
# 공유) 핸들러 등록 순서에 따라 누출될 수 있다. 항상 표준 필드로 취급해 명시적으로 제외한다.
_STANDARD_RECORD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
) | {"message", "asctime"}


class _JsonLinesFormatter(logging.Formatter):
    """LogRecord를 한 줄짜리 JSON으로 직렬화한다."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_ATTRS
        }
        if extras:
            payload["extra"] = extras
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str, config: LoggingConfig | None = None) -> logging.Logger:
    """이름별로 핸들러를 구성하는 로거 팩토리.

    같은 이름 + 같은 설정으로 여러 번 호출하면(모듈을 여러 번 import 하는 상황 등) 기존
    로거를 그대로 재사용해 핸들러가 중복 등록되는 것을 막는다.

    반면 같은 이름에 **다른 설정**이 들어오면 기존 핸들러를 제거하고 새로 구성한다. 이름만
    보고 무조건 재사용하면 로그 경로/레벨 변경이 조용히 무시되어, 호출자가 지정한 위치가
    아닌 이전 경로에 로그가 쌓이는 혼란이 발생한다.
    """
    logger = logging.getLogger(name)
    config = config or LoggingConfig()

    with _CONFIG_LOCK:
        if _CONFIGURED_LOGGERS.get(name) == config:
            return logger

        # 설정이 바뀌었으면 기존 핸들러를 정리한 뒤 재구성한다(파일 핸들러는 닫아 누수 방지).
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        logger.setLevel(config.level)
        logger.propagate = False

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(console_handler)

        config.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            config.log_dir / config.json_lines_filename,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(_JsonLinesFormatter())
        logger.addHandler(file_handler)

        _CONFIGURED_LOGGERS[name] = config

    return logger


@contextmanager
def stage_timer(
    logger: logging.Logger,
    stage: str,
    *,
    truss_id: str | None = None,
    fruit_id: str | None = None,
    **extra_fields: Any,
) -> Iterator[dict[str, Any]]:
    """스테이지 실행 시간을 측정하고 시작/종료/예외를 구조화 로그로 남기는 컨텍스트 매니저.

    사용 예:
        with stage_timer(logger, "stage_a_ripeness", fruit_id=fid) as ctx:
            result = ripeness_gate(...)
            ctx["ripeness_stage"] = result.stage  # 종료 로그에 추가 필드로 포함됨

    시간 측정은 time.perf_counter()를 사용한다(단조 증가 시계라 시스템 시간 보정/DST 등에
    영향받지 않아 성능 계측 용도로 time.time()보다 안전하다).
    """
    context: dict[str, Any] = dict(extra_fields)
    start = time.perf_counter()
    logger.info(
        "stage_start",
        extra=_sanitize_extra(
            {"stage": stage, "truss_id": truss_id, "fruit_id": fruit_id, **extra_fields}
        ),
    )
    try:
        yield context
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.exception(
            "stage_error",
            extra=_sanitize_extra(
                {
                    "stage": stage,
                    "truss_id": truss_id,
                    "fruit_id": fruit_id,
                    "elapsed_ms": round(elapsed_ms, 2),
                    **context,
                }
            ),
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "stage_end",
            extra=_sanitize_extra(
                {
                    "stage": stage,
                    "truss_id": truss_id,
                    "fruit_id": fruit_id,
                    "elapsed_ms": round(elapsed_ms, 2),
                    **context,
                }
            ),
        )
