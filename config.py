import os
import sys
from crewai import LLM

# 윈도우 환경에서 유니코드 출력(이모지 등) 시 cp949 인코딩 오류 방지
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# =====================================================================
# 0. 불필요한 알림 상자 및 로그 차단 & API 키 검증
# =====================================================================
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
    print("\n[오류] GOOGLE_API_KEY가 설정되지 않았습니다.")
    print("👉 '.env' 파일을 열고 올바른 Gemini API 키를 입력해 주세요.\n")
    sys.exit(1)

# =====================================================================
# 1. Gemini 3.1 Flash Lite 모델 설정
# =====================================================================
llm = LLM(
    model="gemini/gemini-3.1-flash-lite",
    temperature=0.3
)
