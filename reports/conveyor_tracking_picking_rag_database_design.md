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
2. **정규화기**는 이벤트 순서, ID 연결, 클록 도메인, 스키마 유효성을 검사한다. 블록 시작 전 **1초 이상에 걸친 30회** 교환으로 `clock_mapping`을 만들고, **매 5분마다 10회** 교환으로 같은 한계를 재확인한다. p95 2 ms·최대 5 ms 조건을 통과한 mapping만 `accepted`로 표시한다. 근거: PRD 1.3절.
3. **분석 저장소**는 원시 이벤트, 결정론적 파생값, 영상 객체 저장소 URI, 수동 주석, 실패 분석 결과를 분리 저장한다. 실패 분류기는 규칙과 연결된 증거만으로 결과를 낸다.
4. **RAG 색인기**는 승인된 안전 SOP, PRD의 안전 규칙, 캘리브레이션·운전 기록, 확정 실패 분석을 청크와 버전 단위로 색인한다. 구조화 저장소의 `trial_id` 기반 증거 묶음은 검색 필터와 인용 대상으로 제공한다.
5. **한국어 질의 서비스**는 질문의 장비·상태·fault 코드·기간·`trial_id`를 추출해 구조화 증거를 먼저 좁힌 뒤 문서를 검색하고, 인용 가능한 근거와 함께 설명하거나 답변을 보류한다. 결과는 읽기 전용이다.

`Pi/ESP32 → 원시 이벤트 저장소 → 정규화·결정론 분석 → 증거 묶음 → RAG 검색·설명` 흐름이며, 반대 방향의 제어 연결은 없다.

## 4. 논리 스키마

모든 테이블은 `id`, `created_at`, `schema_version`, 원본 식별자 또는 해시를 갖는다. 대용량 원본 영상·로그는 객체 저장소에 두고, DB에는 무결성 해시와 불변 URI만 둔다.

| 범주 | 핵심 엔터티와 필드 | 설계 결정 |
| :--- | :--- | :--- |
| 시험·블록·구성 | `experiment`, `block`, `trial`, `config_snapshot`, `calibration` | `block_id`, ON/OFF 조건, 고정 단일 `object_class_id`, 레인 폭·중심, `T_release`, 속도, 조명, 펌웨어·파라미터 해시와 아래 guard snapshot을 고정하고 블록별 실측 온도·속도·압력 진단값을 기록한다. 블록 중 변경은 새 snapshot으로만 남긴다. |
| 원시 Pi/ESP32 이벤트 | `raw_event`, `pi_event`, `esp32_event`, `clock_exchange` | `event_type`, `clock_domain`, 원시 시간, sequence, `trial_id`, `command_id`, payload 원문을 append-only로 보존한다. `clock_exchange`는 매 교환의 `p_send`, `e_rx`, `e_tx`, `p_rx`를 모두 보존한다. |
| 클록 매핑 | `clock_mapping`, `clock_mapping_sample`, `mapping_interlock` | 계수 `a`, `b`, p95·최대 residual, 30회/1초 이상 초기 교환 또는 5분/10회 재확인 구분, 수락·무효화·재수락 시각과 원인을 기록한다. `mapping_interlock`은 실패 시각부터 재수락 시각까지 새 픽이 금지되었음을 Pi 요청·ESP32 상태·명령 거부 이벤트로 증명한다. 파생 시각은 반드시 `mapping_id`를 참조한다. |
| 검출·예측·명령 | `frame`, `detection`, `prediction_plan`, `command`, `command_ack` | `t_exp_start_pi`, `s0`, `y_cam`, `encoder_count0`, `s_est_now`, `L_hat`, `t_contact_pred_pi`, `x_cmd`, 수렴 결과와 NO_PICK 사유를 저장한다. 미래 표본을 사용한 표시는 허용하지 않는다. |
| 안전·모션·진공 | `safety_guard_snapshot`, `safety_transition`, `motion_event`, `vacuum_sample`, `pressure_event`, `limit_event` | 상태 전이 원인과 X/Z 실제 기동·도착, 진공 인가·해제, 압력, soft/physical limit, E-stop·watchdog를 사건 순서대로 남긴다. guard snapshot은 아래의 불변 ESP32 설정을 참조한다. |
| 공급 | `feed_release`, `feed_sensor_event`, `feed_fault` | 계획·실제 방출 시각, 블록 시작에 고정·기록한 `T_release`와 `spacing_tolerance`, 실제 간격이 `T_release ± spacing_tolerance` 안인지, 그리고 `T_release >= T_cycle_p95 + 20%`인지, 중복·미방출, 순환 물체 수와 교체 기록을 보관한다. |
| 기준 영상·주석 | `reference_video`, `video_marker`, `annotation_session`, `annotation`, `reannotation` | 원시 기준 영상, 동기 마커, 블라인드 주석자·세션·순번, `e_contact`, success/failure/unscorable와 제한된 사유, 불일치 원본을 보존한다. 압력·하류 카운터는 진단 신호로만 표시한다. |
| 실패·출처 | `failure_analysis`, `failure_evidence`, `provenance_link` | 결정론적 failure code, 적용 규칙 버전, 분석 입력 해시, 관련 이벤트·영상·주석·문서 구간을 다대다로 연결한다. 분석 결과를 원시 사실처럼 덮어쓰지 않는다. |
| RAG 코퍼스·검색 | `knowledge_document`, `knowledge_revision`, `knowledge_chunk`, `embedding`, `retrieval_run`, `retrieval_citation`, `abstention` | 문서 승인 상태·버전·유효일, 청크 위치, 색인 모델 버전, 질의·필터·점수·인용·보류 사유를 감사 가능하게 남긴다. |

`trial`은 모든 분석의 중심 키다. `raw_event`는 `trial_id`가 없는 초기 부팅·블록 사건도 허용하되, 연결 불가 사유를 남긴다. 주석의 최종 결과는 독립 기준 영상의 사람이 판정한 값이며 자동 압력·카운터로 대체하지 않는다. 근거: PRD 3.1절.

### 블록 고정 ESP32 guard snapshot

`safety_guard_snapshot`은 블록 시작 시 ESP32에 설정한 뒤 블록 중 바꿀 수 없는 다음 값을 원시 설정값과 함께 보존한다. 이는 RAG의 설정 제안이 아니라 분석·감사용 증거다.

- **heartbeat와 stale**: Pi는 100 ms마다 유효 `trial_id` 또는 keepalive를 보내며, 마지막 유효 heartbeat 후 300 ms가 지나면 `READY`·`TRACK`은 새 픽 금지 후 `RECOVER`, `ARMED`·`PICK`·`VERIFY`·`RECOVER`는 즉시 `FAULT_STOP`·모션 취소·진공 해제다. `TRACK`의 후보 프레임은 100 ms 이하만 허용하고, 그 초과 또는 비단조 사건은 `NO_PICK_STALE_CAMERA` 후 `RECOVER`; `ARMED` 이후에는 `FAULT_STOP`이다.
- **encoder**: 10 ms 표본에서 카운트 역행 또는 증가량이 `ceil(1.25 × counts_per_mm × configured_belt_speed_mm_s × 0.010) + 1`보다 크면 이상이다. `TRACK` 전에는 `CALIBRATE`, `ARMED` 이후에는 즉시 `FAULT_STOP`이다.
- **pressure/vacuum**: `P_hold`는 `CALIBRATE`에서 정해 로그한 고정 흡착 임계값이다. `t_vacuum_on_esp` 뒤 200 ms 안에 `P_vac <= P_hold`가 아니면 `PRESSURE_MISS`로 진공 해제·안전 Z 상승·`RECOVER`한다. 최근 10회 `PICK` 중 3번째 miss는 즉시 `FAULT_STOP`; 센서값 50 ms 이상 부재 또는 연속 3개 유효 범위 밖도 즉시 `FAULT_STOP`이다.
- **제한과 전이 원인**: X/Z soft 또는 physical limit은 즉시 모션 중지·진공 해제·`FAULT_STOP`·위치 신뢰 상실 및 재홈 요구다. `safety_transition`은 from/to 상태뿐 아니라 E-stop, timeout, stale, encoder, pressure, limit, no-pick, 복귀 완료 등 촉발 원인과 안전 동작을 기록한다. PRD 6절의 `BOOT`부터 `FAULT_STOP`까지의 전이 이외를 추가 정의하지 않는다.

### 실험·주석 무결성 규칙

`block`은 ON/OFF를 짧은 블록 단위로 무작위 배정해 교차 실행한 `assignment_seed`와 순서를 보존한다. ON/OFF는 보상 항만 제외하고 단일 `object_class_id`, 조명, 레인, 높이, 진공 설정, X/Z 프로파일, 속도, 간격, 게이트 임계값, 캘리브레이션, 안전 여유를 동일하게 고정한다. 블록마다 실측 온도·속도·압력 진단값을 기록해 조건 드리프트를 검토한다. 각 조건은 warm-up **첫 20개 유효 지연 표본**을 KPI·ablation 분모에서 제외한 뒤 독립 판정 가능한 적격 시도 150개 이상을 확보해야 한다. 조건별 `unscorable / (independently_scorable + unscorable)`는 5% 이하여야 하며, 초과 시 추가 표본으로 덮지 않고 No-Go다.

`unscorable` 사유는 기준 영상 파일 부재·손상, 동기 LED/공통 마커 부재로 `trial_id` 연결 불가, 접촉과 빈 도달의 필수 구간에서 물체 또는 흡반이 완전히 가려진 경우로만 제한한다. 압력값, 하류 카운터, 판독 가능한 저화질 영상, 결과가 나빠 보인다는 판단은 사유가 아니다. 주석 목록은 시간·조건순이 아닌 `trial_id` 기준 무작위 순서이며 블라인드로 처리한다. 세션은 최대 60건 또는 60분, 세션 사이 최소 15분 휴식이다. 적격 시도의 최소 20%는 서로 다른 세션에서 재주석하고, `e_contact` 차이를 세션 내 순번에 대해 평가해 유의한 추세가 있으면 해당 구간을 재주석한다. 근거: PRD 3.1·3.3절.

## 5. 결정론적 실패 분석 절차

1. `trial_id`와 `block_id`를 입력으로 원시 이벤트, 설정 snapshot, 수락 mapping, 기준 영상·주석을 고정된 순서로 조회한다.
2. `clock_exchange`의 네 원시 시각과 초기 30회/1초 이상·5분마다 10회 재확인 이력을 검사한다. ID 일치, 시계 단조성, 프레임 수명, encoder 연속성, command ACK 연결, mapping 수락 여부를 검사한다. residual이 p95 2 ms 또는 최대 5 ms를 넘으면 마지막 수락 mapping 이후 시도를 `INVALID_CLOCK_MAPPING`으로 무효화하고, `mapping_interlock`에서 재수락 전 새 픽 요청이 거부되었음을 확인한다. 불충족을 추정값이나 마지막 지연값으로 메우지 않는다.
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
| 1주차 | 사건 스키마, 분리 클록 로그, 지연 사전 측정 | 네 시각 `clock_exchange`, 매핑 무효화·재수락·interlock, config/guard snapshot을 수집한다. 카메라 수신, 검출, 통신, ESP32 기동, X/Z 동작 지연을 분리해 premeasure하고 p95를 기록한다. |
| 2주차 | 기구·인코더·안전 상태 점검 | 불변 ESP32 guard snapshot, 안전 전이 원인, 모션·진공·공급 이벤트와 캘리브레이션 증거를 수집한다. RAG는 아직 연결하지 않는다. |
| 3주차 | 검출·예측·no-pick·간격 측정 | detection/prediction/command와 feed 레코드를 연결하고, `T_release >= T_cycle_p95 + 20%` 및 블록 시작에 고정한 `spacing_tolerance`에 따른 `T_release ± spacing_tolerance` 실제 간격 판정을 측정·기록해 결정론 failure rule을 검증한다. |
| 4주차 | 기준 영상·동기 마커·주석 절차 | 영상 URI·해시, 제한된 unscorable 사유, 세션 분할 블라인드 주석, provenance 연결을 완성한다. |
| 5주차 | ON/OFF interleaved 실행 | assignment seed가 있는 block-randomized ON/OFF 실행으로 warm-up 제외 후 조건당 독립 판정 가능 적격 시도 150개 이상을 축적한다. RAG는 읽기 전용 검토용으로만 시범 검색한다. |
| 6주차 | 수동 주석·재주석·보고 | 20% 교차 세션 재주석과 드리프트 평가, 조건별 5% 이하 unscorable 검사를 마치고, 승인 안전 문서·확정 분석의 인용·보류 감사 로그를 평가한다. |

## 8. 검증 및 수용 체크리스트

- [ ] 모든 `trial_id`가 block, config snapshot, 원시 Pi/ESP32 이벤트, 해당 mapping 또는 연결 불가 사유와 추적 가능하다.
- [ ] `clock_exchange`마다 `p_send/e_rx/e_tx/p_rx`가 보존되고, 블록 시작 1초 이상 30회와 매 5분 10회 재확인이 p95 2 ms·최대 5 ms 기준으로 수락 또는 무효화된다.
- [ ] cross-domain 계산은 수락 `mapping_id`를 참조하며, mapping 실패 시 무효화·재수락 시각과 그 사이 새 픽 거부를 보이는 Pi/ESP32 interlock 증거가 있다.
- [ ] `L_hat_k`는 완료된 과거 유효 표본만 사용하며, 현재·미래 표본과 ACK 귀환 지연을 섞지 않는다.
- [ ] 블록별 불변 ESP32 guard snapshot에 heartbeat/stale, encoder 식, `P_hold`, 200 ms pressure window·10회 중 3회 miss, 센서 window, limit 대응 및 전이 원인이 보존된다.
- [ ] 안전 전이, E-stop, watchdog, limit, 압력, 공급 fault와 NO_PICK 사유가 원시 사건에 연결되고, 분석 결과가 원시 로그를 변경하지 않는다.
- [ ] camera/detection/communication/ESP32-start/X-Z 지연 premeasurement와 `T_cycle_p95 + 20%` 이상인 `T_release`, 블록 고정 `spacing_tolerance`, `T_release ± spacing_tolerance` 실제 간격 판정·편차가 블록에 연결된다.
- [ ] ON/OFF 블록의 고정 단일 `object_class_id`와 나머지 고정 조건이 일치하고, 블록별 실측 온도·속도·압력 진단값이 기록되어 보상 항만 달랐음을 감사할 수 있다.
- [ ] warm-up 20개 제외, block-randomized ON/OFF 배정, 조건당 독립 판정 가능 적격 시도 150개 이상, 조건별 unscorable 5% 이하가 검증된다.
- [ ] 독립 기준 영상·블라인드 주석·제한된 unscorable 사유·세션 정보가 보존되고, 20% 교차 세션 재주석·60건/60분·15분 휴식·무작위화·드리프트 평가가 완료된다. 자동 진단값은 최종 정답을 대체하지 않는다.
- [ ] RAG의 모든 사실 답변에 시험 또는 문서 인용이 있고, 근거 부족·충돌 시 한국어 보류 응답과 사유가 남는다.
- [ ] RAG 서비스 계정에는 읽기 전용 DB·문서 권한만 있으며 Pi/ESP32 명령, 설정 쓰기, fault 해제, 복구 승인 API가 없다.
- [ ] PRD 7절의 여섯 Go/No-Go 게이트를 통과하기 전에는 RAG 설명을 성능·안전 인증 또는 권위 있는 실패 판정으로 주장하지 않는다.

## 참고

- `docs/conveyor_tracking_picking_prd.md` §1.3 시간·사건·클록 도메인
- `docs/conveyor_tracking_picking_prd.md` §3 측정·독립 정답·KPI·ablation
- `docs/conveyor_tracking_picking_prd.md` §6 안전 상태와 실패 처리
- `docs/conveyor_tracking_picking_prd.md` §7 4~6주 로드맵과 MVP Go/No-Go
