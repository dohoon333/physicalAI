import os
from crewai.tools import tool

# =====================================================================
# 2. 실무 부서용 탐색 도구 (Tools)
# =====================================================================
DOCS_DIR = "./docs"

@tool("list_doc_files")
def list_doc_files() -> str:
    """'docs' 폴더 내 마크다운(.md) 참고 파일 목록을 확인합니다."""
    if os.path.exists(DOCS_DIR):
        files = [f for f in os.listdir(DOCS_DIR) if f.endswith('.md')]
        if files:
            return f"docs 폴더 목록: {', '.join(files)}"
        return "docs 폴더에 .md 파일이 없습니다."
    return "docs 폴더가 존재하지 않습니다."

@tool("read_doc_file")
def read_doc_file(file_name: str) -> str:
    """docs 폴더 내 특정 마크다운 파일의 상세 내용을 읽습니다."""
    file_path = os.path.join(DOCS_DIR, file_name)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return f"⚠️ 오류: '{file_name}' 파일이 존재하지 않습니다."

@tool("read_html_template")
def read_html_template() -> str:
    """기존 'index.html' 웹사이트 템플릿 소스코드를 읽어옵니다."""
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "⚠️ index.html 파일이 존재하지 않습니다."

@tool("write_html_template")
def write_html_template(file_content: str) -> str:
    """수정되거나 새로 작성된 'index.html' 웹사이트 소스코드를 디스크에 저장합니다."""
    try:
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(file_content)
        return "Success: index.html 파일이 성공적으로 업데이트되었습니다."
    except Exception as e:
        return f"Error: 파일 저장 중 오류 발생 - {str(e)}"


