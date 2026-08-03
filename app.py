import os
from datetime import datetime
from crewai import Crew, Process
from agents import chief_pm, hardware_team, embedded_control_team, ai_vision_team, biz_pm_team, mlops_web_team
from tasks import project_task

# =====================================================================
# 4. CEO(학생)의 업무 지시 입력 (유니코드 서러게이트 문자 제거 보완)
# =====================================================================
print("\n" + "="*60)
raw_instruction = input("💬 CEO님, 오늘 회사 부서들에게 내릴 업무 지시를 입력하세요:\n> ")
print("="*60 + "\n")

# [보완] 인코딩 예외 방지를 위해 안전한 UTF-8 문자열로 재정제
ceo_instruction = raw_instruction.encode('utf-8', 'ignore').decode('utf-8')


# =====================================================================
# 6. 계층형(Hierarchical) 크루 실행
# =====================================================================
company_crew = Crew(
    agents=[hardware_team, embedded_control_team, ai_vision_team, biz_pm_team, mlops_web_team],
    tasks=[project_task],
    process=Process.hierarchical,
    manager_agent=chief_pm,
    verbose=False
)

print("⏳ PM이 지시사항을 분석하고 필요한 부서를 호출하여 업무를 수행 중입니다...\n")

# [보완] kickoff 호출 시 inputs 딕셔너리로 안전하게 넘겨줍니다.
result = company_crew.kickoff(
    inputs={'ceo_instruction': ceo_instruction}
)

# =====================================================================
# 7. 최종 보고서 출력 및 저장
# =====================================================================
print("\n" + "="*60)
print("📋 [CEO 제출용 최종 업무 완료 보고서]")
print("="*60 + "\n")
print(result)

REPORTS_DIR = "./reports"
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filepath = os.path.join(REPORTS_DIR, f"report_{timestamp}.md")

with open(report_filepath, "w", encoding="utf-8") as f:
    f.write(str(result))

print(f"\n📁 보고서 파일 저장 완료: '{report_filepath}'\n")