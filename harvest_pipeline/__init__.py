"""방울토마토 자동 수확 로봇 — 데이터 전처리 파이프라인 패키지.

설계 근거 문서:
- docs/robot_arm_pick_and_place_prd.md (PRD v2)
- docs/cherry_tomato_harvest_field_survey_checklist.md
- docs/cherry_tomato_harvest_preprocessing_pipeline.md (전처리 파이프라인 v2)

패키지 구조:
- config.py            : 파라미터 스키마 및 YAML 로더 (모든 임계값은 여기서만 정의)
- exceptions.py         : 예외 계층 (진짜 오류만 예외로, 정상 분기는 결과 객체로 표현)
- logging_utils.py      : 구조화 로깅(JSON) 설정
- interfaces.py         : 교체 가능한 AI 모델 컴포넌트의 Protocol 정의
- stage0_common.py      : Stage 0 공통 전처리
- stage_a_ripeness.py   : Stage A 숙성도 판별 게이트
- stage_b_grasp_cut.py  : Stage B 파지·절단 결합 포즈 추정
- pipeline.py           : 전체 오케스트레이션
"""

__version__ = "2.0.0"
