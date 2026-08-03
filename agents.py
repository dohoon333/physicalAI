from crewai import Agent
from config import llm
from tools import list_doc_files, read_doc_file, read_html_template, write_html_template

# =====================================================================
# 3. 부서별 페르소나 및 PM 매니저 정의
# =====================================================================
tools_list = [list_doc_files, read_doc_file]

chief_pm = Agent(
    role='총괄 PM (Chief Project Manager)',
    goal='CEO의 업무 지시를 분석하여, 해당 지시를 수행하는 데 꼭 필요한 하위 부서(에이전트)에게만 업무를 할당하고 그 결과를 종합하여 보고한다.',
    backstory='스타트업 및 기술 기업의 오퍼레이션 총괄 책임자로, 불필요한 공수를 줄이고 필요한 부서를 적재적소에 배치하는 최고의 리더.',
    llm=llm,
    verbose=True
)

hardware_team = Agent(
    role='하드웨어 & 메카트로닉스 팀',
    goal='기계 구조, 센서 배치, 케이싱 설계 등 물리적 하드웨어 관련 업무를 수행한다.',
    backstory='기계 기구 설계, 센서 융합 및 물리 환경 대응 전문가.',
    tools=tools_list,
    llm=llm,
    verbose=True
)

embedded_control_team = Agent(
    role='임베디드 & 로봇 제어 팀',
    goal='MCU 펌웨어, ROS2 로봇 제어, 통신 아키텍처 및 모터 제어 관련 업무를 수행한다.',
    backstory='MCU, ROS2, 산업용 통신 및 모터 제어 전문가.',
    tools=tools_list,
    llm=llm,
    verbose=True
)

ai_vision_team = Agent(
    role='AI & 데이터 파이프라인 팀',
    goal='컴퓨터 비전, AI 모델 학습, 데이터 전처리 및 ONNX 경량화 추론 최적화 업무를 수행한다.',
    backstory='컴퓨터 비전, AI 모델 학습, ONNX/TensorRT 경량화 전문가.',
    tools=tools_list,
    llm=llm,
    verbose=True
)

biz_pm_team = Agent(
    role='비즈니스 & 프로덕트 기획 팀',
    goal='제품 PRD 작성, 사업성 평가, ROI 산출, 사용자 페인포인트 분석 업무를 수행한다.',
    backstory='IT/AI 제품 기획 및 사업성/ROI 분석 전문가.',
    tools=tools_list,
    llm=llm,
    verbose=True
)

mlops_web_team = Agent(
    role='웹 개발 & MLOps 팀',
    goal='웹 프론트엔드/백엔드 소스코드 작성, 로그인/인증 기능, 시스템 통합 및 모니터링 화면 구축 업무를 수행한다.',
    backstory='풀스택 웹 개발, FastAPI 백엔드, DevOps/MLOps 전문가.',
    tools=[list_doc_files, read_doc_file, read_html_template, write_html_template],
    llm=llm,
    verbose=True
)
