"""
schema.py — 모든 자료의 공통 레코드 구조.

핵심 원칙: 출처 태그 없는 데이터는 시스템에 들어올 수 없다.
레이어4의 해설이 "출처를 밝혀서" 나오려면, 애초에 모든 원본이
어디서 왔는지를 들고 있어야 한다.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import hashlib


# 허용되는 값들 (오타·혼란 방지용 화이트리스트)
LAYERS = {"L1_pattern", "L2_corpus", "L3_trend", "L4_generate", "L5_exam"}
LEVELS = {"초등", "중등", "특수", "공통"}   # 급별 (기출 교차분석용)
SUBJECTS = {
    "총론", "창의적체험활동", "통합교과",
    "국어", "영어", "수학", "사회", "과학",
    "미술", "음악", "체육", "실과", "도덕",
    "공통",   # 교육과정 총론처럼 과목에 안 매이는 자료
}

# 자료종류 — 네 가지가 성격이 다름(가장 중요한 축)
DOC_TYPES = {
    "교육과정_총론",     # 전 교과 공통(과목=공통). 상위 원리·역량·인간상
    "교육과정_성취기준",  # 과목별. 코드 있음
    "지도서_총론",       # 과목별. 그 과목 교수법·평가
    "지도서_각론",       # 과목별. 단원별 개념(개념의 축)
}


@dataclass
class Record:
    """
    자료 한 조각(문장/문단/문항 하나)의 최소 단위.

    필수: text, layer, subject, source
    자료 구조 축(선택, L2 자료의 정확한 분류용):
      - doc_type   : 교육과정_총론 / 교육과정_성취기준 / 지도서_총론 / 지도서_각론
      - concepts   : 개념 태그 리스트(각론의 핵심). 각론=개념 원천, 총론/교육과정=응용
      - grade_band : 학년군(3~4학년군 등)
      - area       : 영역(읽기/쓰기/문법/문학/듣기말하기 등)
      - unit       : 단원명(각론, 자료에서 추출)
      - model      : 학습모형(반응중심/직접교수법 등)
    기출/문제 축:
      - year, level, code, qtype
      - exam_type  : 수능형(기타형) 문제의 유형 태그
    """
    text: str
    layer: str
    subject: str
    source: str                      # ← 출처. 무조건 채워야 함.
    year: Optional[int] = None
    level: Optional[str] = None
    code: Optional[str] = None
    qtype: Optional[str] = None
    doc_type: Optional[str] = None
    concepts: Optional[list] = None
    grade_band: Optional[str] = None
    area: Optional[str] = None
    unit: Optional[str] = None
    model: Optional[str] = None
    exam_type: Optional[str] = None
    rec_id: str = field(default="")

    def __post_init__(self):
        # --- 출처 태그 강제 ---
        if not self.text or not self.text.strip():
            raise ValueError("Record.text 가 비어있음")
        if not self.source or not self.source.strip():
            raise ValueError(f"출처(source) 없는 데이터는 거부됨: {self.text[:30]!r}")
        if self.layer not in LAYERS:
            raise ValueError(f"알 수 없는 layer={self.layer!r} (허용:{LAYERS})")
        if self.subject not in SUBJECTS:
            raise ValueError(f"알 수 없는 subject={self.subject!r} (허용:{SUBJECTS})")
        if self.doc_type is not None and self.doc_type not in DOC_TYPES:
            raise ValueError(f"알 수 없는 doc_type={self.doc_type!r} (허용:{DOC_TYPES})")
        if self.concepts is None:
            self.concepts = []
        if self.level is not None and self.level not in LEVELS:
            raise ValueError(f"알 수 없는 level={self.level!r} (허용:{LEVELS})")
        # L1(기출 패턴)은 연도가 반드시 있어야 시계열 분석 가능
        if self.layer == "L1_pattern" and self.year is None:
            raise ValueError("L1_pattern 레코드는 year(출제연도)가 필수")
        # 안정적 id (같은 내용+출처면 같은 id → 중복 방지)
        if not self.rec_id:
            h = hashlib.md5(f"{self.source}|{self.text}".encode("utf-8")).hexdigest()[:12]
            self.rec_id = h

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Record":
        return Record(**d)


def save_records(records: list[Record], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, ensure_ascii=False, indent=2)


def load_records(path: str) -> list[Record]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Record.from_dict(d) for d in data]


# ── pkl 저장/로드 (Record 객체 그대로 보존) ──────────────────────
import pickle as _pickle

def save_records_pkl(records, path):
    """Record 리스트를 pkl로 저장(사용자 원본 자료용)."""
    with open(path, "wb") as f:
        _pickle.dump([r.to_dict() for r in records], f)
    # 서버 백업: pkl 최신본 + DB 'records' 테이블 미러링
    try:
        import cloud
        cloud.push(path)
        cloud.sync_records(path, records)
    except ImportError:
        pass

def load_records_pkl(path):
    """pkl에서 Record 리스트 복원. 없으면 빈 리스트."""
    import os
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = _pickle.load(f)
    return [Record.from_dict(d) for d in data]
