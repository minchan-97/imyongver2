"""
paths.py — 과목별 데이터/산출물 pkl 경로를 한 곳에서 관리.

저장 구조 (data/ 아래):
  원본 자료:  {과목}_L2.pkl          (성취기준·지도서 등)
              {과목}_L1.pkl          (기출: 연도·급별 태그)
              {과목}_L3.pkl          (검색으로 모은 트렌드 자료)
  학습 산출물:{과목}_emb.pkl         (고정 임베딩)
              {과목}_som.pkl         (과목 SOM)

과목만 바꾸면 다른 과목으로 그대로 복제된다.
"""
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(BASE, exist_ok=True)

def _p(name):
    return os.path.join(BASE, name)

def l2_path(subject):  return _p(f"{subject}_L2.pkl")
def l1_path(subject):  return _p(f"{subject}_L1.pkl")
def l3_path(subject):  return _p(f"{subject}_L3.pkl")
def emb_path(subject): return _p(f"{subject}_emb.pkl")
def som_path(subject): return _p(f"{subject}_som.pkl")

def exists(path):
    return os.path.exists(path)

def study_path(subject): return _p(f"{subject}_study.pkl")

def concept_dict_path(subject): return _p(f"{subject}_concepts.pkl")  # 개념 태그 사전
def exam_path(subject):         return _p(f"{subject}_L5_exam.pkl")    # 수능형(기타형) 문제
def common_chongron_path():     return _p("공통_교육과정총론.pkl")      # 전 과목 공유

def exam_weak_path(subject): return _p(f"{subject}_exam_weak.pkl")  # 수능형 유형별 약점

def passage_path(subject): return _p(f"{subject}_passages.pkl")  # 기출 지문 세트


def all_paths(subject):
    """서버 동기화 대상 전체 (이 과목 pkl + 공통 총론)."""
    return [l2_path(subject), l1_path(subject), l3_path(subject),
            emb_path(subject), som_path(subject), study_path(subject),
            concept_dict_path(subject), exam_path(subject),
            exam_weak_path(subject), passage_path(subject),
            common_chongron_path(), scan_cache_path(), drive_state_path(),
            _p(f"trendlab_{subject}.pkl"), _p(f"labels_{subject}.pkl"),
            _p(f"tagger_{subject}.pkl"), _p(f"health_{subject}.pkl")]


def scan_cache_path(): return _p("scan_cache.pkl")   # 페이지 스캔 결과 캐시(전 과목 공용)


def drive_state_path(): return _p("drive_state.pkl")  # 드라이브에서 가져온 파일 기록


def discover_subjects(extra_names=None):
    """
    자료가 있는 과목 찾기. 로컬 data/ 의 파일명 + (있으면) 서버 파일명에서
    '{과목}_L1.pkl' / '{과목}_L2.pkl' 꼴을 골라낸다.
    """
    from schema import SUBJECTS
    names = list(extra_names or [])
    if os.path.isdir(BASE):
        names += os.listdir(BASE)
    found = set()
    for n in names:
        for s in SUBJECTS:
            if n in (f"{s}_L1.pkl", f"{s}_L2.pkl"):
                found.add(s)
    return sorted(found)
