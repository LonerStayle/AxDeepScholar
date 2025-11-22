from datetime import datetime
def get_today_str(pattern = "%a %b %-d, %Y"):
    return datetime.now().strftime(pattern)

from pathlib import Path
def get_current_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()
    
def get_project_root(marker="pyproject.toml"): # 현재 프로젝트의 경로를 가져오기 위해 사용용
    cur = Path(__file__).resolve() if "__file__" in globals() else Path().resolve()
    return next((p for p in [cur, *cur.parents] if (p / marker).exists()), cur)
    

    
# import os, sys
# from pathlib import Path
# src_path = Path(os.getcwd()).resolve().parents[1]  
# sys.path.append(str(src_path))