"""
exam_practice.py — 수능형(기타형) 문제 순수 연습.

성격: 출처 없이 제시문 해석해 바로 푸는 문제. 개념 지도에 매핑하지 않음.
용도(합의): 용도1 — 순수 연습. 풀고 맞았는지 확정, '유형별' 약점만 쌓는다.

저장: {과목}_L5_exam.pkl (문제 은행)
약점: study_state에 exam_type 단위로 기록(개념영역 node 대신).
유형은 미리 못 박지 않고, 사용자가 풀면서 스스로 붙인다(자유 태그).
"""
from __future__ import annotations
CORE_VERSION = "13.3"
import os, pickle, time

try:
    from cloud import push as _cloud_push
except Exception:  # cloud 계층이 없어도 로컬 동작은 유지
    def _cloud_push(path, **kw): return False



class ExamBank:
    """수능형 문제 은행. 문제+정답+해설+유형태그."""
    def __init__(self, subject):
        self.subject = subject
        self.items = []   # {id, question, answer, explanation, exam_type, source}

    def add(self, question, answer="", explanation="", exam_type="", source="수능형 자작"):
        import hashlib
        qid = hashlib.md5(f"{question}".encode()).hexdigest()[:10]
        if any(it["id"] == qid for it in self.items):
            return False
        self.items.append({
            "id": qid, "question": question, "answer": answer,
            "explanation": explanation, "exam_type": exam_type, "source": source,
        })
        return True

    def types(self):
        return sorted(set(it["exam_type"] for it in self.items if it["exam_type"]))

    def by_type(self, exam_type):
        return [it for it in self.items if it["exam_type"] == exam_type]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"subject": self.subject, "items": self.items}, f)
        _cloud_push(path)

    @staticmethod
    def load(path, subject):
        b = ExamBank(subject)
        if os.path.exists(path):
            with open(path, "rb") as f:
                d = pickle.load(f)
            b.items = d.get("items", [])
        return b


class ExamWeakness:
    """수능형 유형별 약점(개념 지도와 독립)."""
    def __init__(self):
        self.type_stats = {}   # exam_type -> [correct, total]

    def record(self, exam_type, correct: bool):
        key = exam_type or "(미분류)"
        c, t = self.type_stats.get(key, [0, 0])
        self.type_stats[key] = [c + (1 if correct else 0), t + 1]

    def weak_types(self):
        out = []
        for t, (c, tot) in self.type_stats.items():
            out.append({"유형": t, "정답": c, "시도": tot,
                        "정답률": round(c / tot, 2) if tot else 0})
        out.sort(key=lambda d: (d["정답률"], -d["시도"]))
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.type_stats, f)
        _cloud_push(path)

    @staticmethod
    def load(path):
        w = ExamWeakness()
        if os.path.exists(path):
            with open(path, "rb") as f:
                w.type_stats = pickle.load(f)
        return w

