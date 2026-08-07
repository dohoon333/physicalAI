# 컨베이어 트래킹 픽킹 RAG·데이터베이스 설계 제안

## 1. 목적, 범위, 제외 범위

이 문서는 지연 보상 X/Z 컨베이어 트래킹 픽킹 연구 셀의 실패·오류 데이터를 수집하고, 안전 운전 지식을 검색하여 **왜 해당 시도가 실패했는지 증거와 함께 설명**하기 위한 제안 설계다. 연구 목표, 물리 범위, KPI 및 안전 권한은 `docs/conveyor_tracking_picking_prd.md`의 1, 3, 5, 6절을 따른다.

- **범위**: 실험 블록, 원시 Pi·ESP32 사건, 클록 매핑, 검출·예측·명령, 안전·모션·진공·공급, 기준 영상 주석, 실패 판정과 출처를 묶어 재현 가능한 분석 기록을 만든다. 승인된 안전 절차와 장애 대응 문서는 한국어 RAG 검색 대상으로 관리한다.
- **제외**: RAG 구현, 실시간 제어 경로 변경, 자동 복구, 상용 안전 인증, 다중 물체·다중 레인 확장이다. 현재 CrewAI 문서 읽기 도구가 있어도 RAG 연동은 아직 제안일 뿐 구현 상태가 아니다.

## 2. 최우선 원칙과 안전 경계

### 권위 있는 구조화 데이터 규칙

시험 사실의 단일 권위 출처는 불변 원시 이벤트와 그로부터 결정적으로 산출한 구조화 레코드다. `trial_id`, 원시 타임스탬프, 수락된 `mapping_id`, 명령 ID, 영상·주석 ID가 연결되지 않은 문장형 요약이나 RAG 답변은 사실 판정 근거가 될 수 없다. 원시 Pi·ESP32 시간을 직접 빼지 않고, 블록별 수락 affine mapping으로 환산한 파생 시각만 교차 도메인 분석에 쓴다. 이는 PRD 1.3절의 residual 수락 기준과 인과성 규칙을 보존한다.

RAG는 **증거를 검색하고 설명할 뿐**이다. RAG는 실패를 권위 있게 분류할 수 없고, 명령 전송, 파라미터 변경, fault 해제, 복구 승인도 할 수 없다. Pi/ESP32의 실시간 제어·heartbeat·E-stop·watchdog 경로에는 배치하지 않는다. ESP32는 계속 안전 상태 기계를 소유하고 Pi는 요청만 한다. `FAULT_STOP` 해제는 수동 원인 확인과 재캘리브레이션 뒤에만 가능하다. 근거: PRD 6절.

## 3. 구성 요소와 데이터 흐름

1. **수집기**는 Pi와 ESP32에서 append-only 원시 이벤트를 수집하고 장치·펌웨어·스키마 버전을 함께 저장한다. 제어 송신과 독립된 비동기 버퍼가 지연 또는 장애여도 제어를 막지 않으며, 수집 누락은 별도 오류로 남긴다.
2. **정규화기**는 이벤트 순서, ID 연결, 클록 도메인, 스키마 유효성을 검사한다. 블록 시작 30회와 5분 재확인의 교환 표본으로 `clock_mapping`을 만들고, p95 2 ms·최대 5 ms 조건을 통과한 mapping만 `accepted`로 표시한다. 근거: PRD 1.3절.
3. **분석 저장소**는 원시 이벤트, 결정론적 파생값, 영상 객체 저장소 URI, 수동 주석, 실패 분석 결과를 분리 저장한다. 실패 분류기는 규칙과 연결된 증거만으로 결과를 낸다.
4. **RAG 색인기**는 승인된 안전 SOP, PRD의 안전 규칙, 캘리브레이션·운전 기록, 확정 실패 분석을 청크와 버전 단위로 색인한다. 구조화 저장소의 `trial_id` 기반 증거 묶음은 검색 필터와 인용 대상으로 제공한다.
5. **한국어 질의 서비스**는 질문의 장비·상태·fault 코드·기간·`trial_id`를 추출해 구조화 증거를 먼저 좁힌 뒤 문서를 검색하고, 인용 가능한 근거와 함께 설명하거나 답변을 보류한다. 결과는 읽기 전용이다.

`Pi/ESP32 → 원시 이벤트 저장소 → 정규화·결정론 분석 → 증거 묶음 → RAG 검색·설명` 흐름이며, 반대 방향의 제어 연결은 없다.

## 4. 논리 스키마

모든 테이블은 `id`, `created_at`, `schema_version`, 원본 식별자 또는 해시를 갖는다. 대용량 원본 영상·로그는 객체 저장소에 두고, DB에는 무결성 해시와 불변 URI만 둔다.

| 범주 | 핵심 엔터티와 필드 | 설계 결정 |
| :--- | :--- | :--- |
| 시험·블록·구성 | `experiment`, `block`, `trial`, `config_snapshot`, `calibration` | `block_id`, ON/OFF 조건, 레인 폭·중심, `T_release`, 속도, 조명, 펌웨어·파라미터 해시를 고정한다. 블록 중 변경은 새 snapshot으로만 남긴다. |
| 원시 Pi/ESP32 이벤트 | `raw_event`, `pi_event`, `esp32_event`, `clock_exchange` | `event_type`, `clock_domain`, 원시 시간, sequence, `trial_id`, `command_id`, payload 원문을 append-only로 보존한다. |
| 클록 매핑 | `clock_mapping`, `clock_mapping_sample` | 계수 `a`, `b`, p95·최대 residual, 수락 여부, 유효 구간을 기록한다. 파생 시각은 반드시 `mapping_id`를 참조한다. |
| 검출·예측·명령 | `frame`, `detection`, `prediction_plan`, `command`, `command_ack` | `t_exp_start_pi`, `s0`, `y_cam`, `encoder_count0`, `s_est_now`, `L_hat`, `t_contact_pred_pi`, `x_cmd`, 수렴 결과와 NO_PICK 사유를 저장한다. 미래 표본을 사용한 표시는 허용하지 않는다. |
| 안전·모션·진공 | `safety_transition`, `motion_event`, `vacuum_sample`, `pressure_event`, `limit_event` | 상태 전이, X/Z 실제 기동·도착, 진공 인가·해제, 압력, soft/physical limit, E-stop·watchdog를 사건 순서대로 남긴다. |
| 공급 | `feed_release`, `feed_sensor_event`, `feed_fault` | 계획·실제 방출 시각, 실제 간격, 중복·미방출, 순환 물체 수와 교체 기록을 보관한다. |
| 기준 영상·주석 | `reference_video`, `video_marker`, `annotation_session`, `annotation`, `reannotation` | 원시 기준 영상, 동기 마커, 블라인드 주석자·세션·순번, `e_contact`, success/failure/unscorable와 사유, 불일치 원본을 보존한다. 압력·하류 카운터는 진단 신호로만 표시한다. |
| 실패·출처 | `failure_analysis`, `failure_evidence`, `provenance_link` | 결정론적 failure code, 적용 규칙 버전, 분석 입력 해시, 관련 이벤트·영상·주석·문서 구간을 다대다로 연결한다. 분석 결과를 원시 사실처럼 덮어쓰지 않는다. |
| RAG 코퍼스·검색 | `knowledge_document`, `knowledge_revision`, `knowledge_chunk`, `embedding`, `retrieval_run`, `retrieval_citation`, `abstention` | 문서 승인 상태·버전·유효일, 청크 위치, 색인 모델 버전, 질의·필터·점수·인용·보류 사유를 감사 가능하게 남긴다. |

`trial`은 모든 분석의 중심 키다. `raw_event`는 `trial_id`가 없는 초기 부팅·블록 사건도 허용하되, 연결 불가 사유를 남긴다. 주석의 최종 결과는 독립 기준 영상의 사람이 판정한 값이며 자동 압력·카운터로 대체하지 않는다. 근거: PRD 3.1절.

## 5. 결정론적 실패 분석 절차

1. `trial_id`와 `block_id`를 입력으로 원시 이벤트, 설정 snapshot, 수락 mapping, 기준 영상·주석을 고정된 순서로 조회한다.
2. ID 일치, 시계 단조성, 프레임 수명, encoder 연속성, command ACK 연결, mapping 수락 여부를 검사한다. 불충족 시 PRD가 정한 `INVALID_TIMESTAMP`, `INVALID_CLOCK_MAPPING` 등으로 우선 기록하고 추정값으로 메우지 않는다.
3. 수락된 mapping만 사용해 `t_x_start_pi_est`, `L_k`, 모션·진공 순서를 계산한다. `L_hat_k`가 과거 유효 표본만 사용했는지 검증하고, 위반이면 분석 불가 또는 규칙 위반으로 표시한다.
4. 안전 상태 전이와 가드를 재생해 `NO_PICK_LANE`, `NO_PICK_STALE_CAMERA`, `NO_PICK_MOTION_LATE`, `NO_PICK_SPACING`, `FEED_FAULT`, `PRESSURE_MISS`, 제한 위반, `FAULT_STOP`의 직접 증거를 연결한다. PRD 2.1절과 6절의 정의 밖 새로운 failure code는 만들지 않는다.
5. 독립 주석이 있으면 접촉 오차와 success/failure를 연결한다. 없거나 `unscorable`이면 자동 진단과 분리해 판정 불가로 남긴다.
6. 규칙 버전, 입력 이벤트 해시, 사용한 mapping·주석·영상 ID, 배제 사유를 `failure_analysis`와 `failure_evidence`에 기록한다. 같은 입력은 같은 결과를 내야 한다.
7. 그 후에만 RAG가 분석 결과와 안전 문서를 읽어 사람이 이해할 설명을 만든다. RAG 설명은 원인 후보, 증거, 안전상 다음 확인 항목을 구분해 제시한다.

## 6. 한국어 RAG 응답·인용·보류 정책

- 답변은 한국어로 작성하고, 각 사실 문장에 `[시험: trial_id / 이벤트: event_id]`, `[문서: 경로 §절 / 개정]` 형식의 인용을 붙인다. 예: `[문서: docs/conveyor_tracking_picking_prd.md §6]`.
- 질의가 특정 시험이면 해당 `trial_id`의 결정론 분석과 증거 링크를 우선한다. 일반 안전 질의면 `approved` 상태의 SOP와 PRD만 검색한다. 초안, 만료본, 권한 없는 문서는 답변 근거에서 제외한다.
- 검색 점수가 낮음, 인용 가능한 문서 부재, 사건 연결 불가, mapping 미수락, 독립 정답 부재, 또는 서로 충돌하는 근거가 있으면 결론을 만들지 않는다. “확인 가능한 근거가 부족하여 원인을 판정할 수 없습니다”라고 보류하고, 필요한 `trial_id`, 원시 이벤트, 영상·주석 또는 승인 절차를 명시한다.
- RAG는 “권장 확인 절차”만 설명한다. `FAULT_STOP` 해제, 재시작, 파라미터 변경, 명령 전송, 안전 복구 승인은 명시적으로 거절하고 ESP32 상태 규칙과 현장 수동 확인으로 안내한다.

## 7. PRD 주차별 도입 계획

| 주차 | PRD 산출물과 연계 | 이 설계의 도입 범위 |
| :--- | :--- | :--- |
| 1주차 | 사건 스키마, 분리 클록 로그, 지연 사전 측정 | 원시 이벤트·clock mapping·config snapshot 스키마와 수락 검사만 구축한다. |
| 2주차 | 기구·인코더·안전 상태 점검 | 안전·모션·진공·공급 이벤트와 캘리브레이션 증거를 수집한다. RAG는 아직 연결하지 않는다. |
| 3주차 | 검출·예측·no-pick·간격 측정 | detection/prediction/command와 feed 레코드를 연결하고 결정론 failure rule을 검증한다. |
| 4주차 | 기준 영상·동기 마커·주석 절차 | 영상 URI·해시, 세션 분할 블라인드 주석, provenance 연결을 완성한다. |
| 5주차 | ON/OFF interleaved 실행 | 고정 조건·블록·trial 증거 묶음과 failure 분석을 축적한다. RAG는 읽기 전용 검토용으로만 시범 검색한다. |
| 6주차 | 수동 주석·재주석·보고 | 승인 안전 문서와 확정 분석을 색인하고, 인용·보류 감사 로그를 포함한 사후 설명을 평가한다. |

## 8. 검증 및 수용 체크리스트

- [ ] 모든 `trial_id`가 block, config snapshot, 원시 Pi/ESP32 이벤트, 해당 mapping 또는 연결 불가 사유와 추적 가능하다.
- [ ] cross-domain 계산은 수락 `mapping_id`를 참조하며, p95 2 ms·최대 5 ms 초과 블록은 `INVALID_CLOCK_MAPPING`과 No-Go로 남는다.
- [ ] `L_hat_k`는 완료된 과거 유효 표본만 사용하며, 현재·미래 표본과 ACK 귀환 지연을 섞지 않는다.
- [ ] 안전 전이, E-stop, watchdog, limit, 압력, 공급 fault와 NO_PICK 사유가 원시 사건에 연결되고, 분석 결과가 원시 로그를 변경하지 않는다.
- [ ] 독립 기준 영상·블라인드 주석·재주석 세션 정보가 보존되고, 자동 진단값이 최종 정답을 대체하지 않는다.
- [ ] RAG의 모든 사실 답변에 시험 또는 문서 인용이 있고, 근거 부족·충돌 시 한국어 보류 응답과 사유가 남는다.
- [ ] RAG 서비스 계정에는 읽기 전용 DB·문서 권한만 있으며 Pi/ESP32 명령, 설정 쓰기, fault 해제, 복구 승인 API가 없다.
- [ ] PRD 7절의 여섯 Go/No-Go 게이트를 통과하기 전에는 RAG 설명을 성능·안전 인증 또는 권위 있는 실패 판정으로 주장하지 않는다.

## 참고

- `docs/conveyor_tracking_picking_prd.md` §1.3 시간·사건·클록 도메인
- `docs/conveyor_tracking_picking_prd.md` §3 측정·독립 정답·KPI·ablation
- `docs/conveyor_tracking_picking_prd.md` §6 안전 상태와 실패 처리
- `docs/conveyor_tracking_picking_prd.md` §7 4~6주 로드맵과 MVP Go/No-Go
