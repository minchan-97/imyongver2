"""
study_state.py — '나와 함께 크는' 학습 상태 관리.

세 가지를 pkl로 누적한다(과목별):
  1) 내 답 채점 이력 → 약점 지도(개념영역 + 성취기준코드 단위)
  2) 문제/해설 검토 피드백(맞다/아니다) → 자료 신뢰도 조정
  3) 신뢰도 조정 이력 → 복구(undo) 가능

핵심 원칙(합의):
  - 자동 채점(LLM/정합도)은 '제안'일 뿐. 약점지도에 실제 반영되는 건
    사용자가 '확정'한 판정(TRUST_HUMAN=1.0). 자동값이 지도를 오염시키지 않음.
  - 검토 페널티는 되돌릴 수 있어야 한다(좋은 자료가 잘못 묻히지 않게).
"""
from __future__ import annotations
CORE_VERSION = "13.3"
import os, pickle, time
from collections import defaultdict

try:
    from cloud import push as _cloud_push
except Exception:  # cloud 계층이 없어도 로컬 동작은 유지
    def _cloud_push(path, **kw): return False



class StudyState:
    def __init__(self, subject):
        self.subject = subject
        # 약점지도: 개념영역(노드)별, 코드별 (정답수, 시도수)
        self.node_stats = defaultdict(lambda: [0, 0])   # node -> [correct, total]
        self.code_stats = defaultdict(lambda: [0, 0])   # code -> [correct, total]
        # 채점 이력 (확정된 것만 지도에 반영)
        self.answer_log = []   # {ts, node, codes, my_answer, auto_suggest, final_verdict}
        # 자료 신뢰도: rec_id -> 가중치(1.0 기본). 낮을수록 문제생성에서 덜 쓰임
        self.trust = defaultdict(lambda: 1.0)
        # 검토/신뢰도 변경 이력 (복구용 스택)
        self.review_log = []   # {ts, rec_id, delta, reason, undone(bool)}

    # ── 1) 답 채점 확정 → 약점지도 반영 ──────────────────────
    def record_answer(self, node, codes, my_answer, auto_suggest, final_verdict):
        """
        final_verdict: 'correct' | 'wrong'  ← 사용자가 확정한 값(이것만 지도 반영)
        auto_suggest : LLM/정합도가 제안한 값(참고 기록용)
        """
        is_correct = (final_verdict == "correct")
        self.node_stats[node][1] += 1
        if is_correct:
            self.node_stats[node][0] += 1
        for c in (codes or []):
            self.code_stats[c][1] += 1
            if is_correct:
                self.code_stats[c][0] += 1
        self.answer_log.append({
            "ts": time.time(), "node": node, "codes": list(codes or []),
            "my_answer": my_answer, "auto_suggest": auto_suggest,
            "final_verdict": final_verdict,
        })

    def weak_by_node(self, min_total=1):
        """개념영역별 약점(정답률 낮은 순)."""
        out = []
        for node, (c, t) in self.node_stats.items():
            if t >= min_total:
                out.append({"node": node, "correct": c, "total": t,
                            "rate": round(c / t, 2)})
        out.sort(key=lambda d: (d["rate"], -d["total"]))
        return out

    def weak_by_code(self, min_total=1):
        """성취기준 코드별 약점."""
        out = []
        for code, (c, t) in self.code_stats.items():
            if t >= min_total:
                out.append({"code": code, "correct": c, "total": t,
                            "rate": round(c / t, 2)})
        out.sort(key=lambda d: (d["rate"], -d["total"]))
        return out

    def weak_nodes_for_targeting(self, top=5):
        """적응형 출제용: 약한 개념영역 노드 우선순위."""
        return [w["node"] for w in self.weak_by_node()[:top]]

    # ── 2) 검토 피드백 → 자료 신뢰도 조정 (복구 가능) ──────────
    def review_feedback(self, rec_id, good: bool, reason=""):
        """
        good=False(이 자료가 나쁜 문제/해설을 만듦) → 신뢰도 하향
        good=True(좋음) → 신뢰도 소폭 상향(최대 1.0)
        """
        delta = -0.3 if not good else +0.1
        old = self.trust[rec_id]
        new = max(0.0, min(1.0, old + delta))
        self.trust[rec_id] = new
        self.review_log.append({
            "ts": time.time(), "rec_id": rec_id, "delta": new - old,
            "reason": reason, "undone": False,
        })
        return new

    def undo_last_review(self):
        """가장 최근의 (되돌리지 않은) 신뢰도 변경을 복구."""
        for entry in reversed(self.review_log):
            if not entry["undone"]:
                self.trust[entry["rec_id"]] -= entry["delta"]
                self.trust[entry["rec_id"]] = max(0.0, min(1.0, self.trust[entry["rec_id"]]))
                entry["undone"] = True
                return entry
        return None

    def undo_review_for(self, rec_id):
        """특정 자료의 마지막 변경 복구."""
        for entry in reversed(self.review_log):
            if entry["rec_id"] == rec_id and not entry["undone"]:
                self.trust[rec_id] -= entry["delta"]
                self.trust[rec_id] = max(0.0, min(1.0, self.trust[rec_id]))
                entry["undone"] = True
                return entry
        return None

    def trust_of(self, rec_id):
        return self.trust[rec_id]

    # ── 저장/로드 ────────────────────────────────────────────
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "subject": self.subject,
                "node_stats": dict(self.node_stats),
                "code_stats": dict(self.code_stats),
                "answer_log": self.answer_log,
                "trust": dict(self.trust),
                "review_log": self.review_log,
            }, f)
        _cloud_push(path)

    @staticmethod
    def load(path, subject):
        s = StudyState(subject)
        if not os.path.exists(path):
            return s
        with open(path, "rb") as f:
            d = pickle.load(f)
        s.node_stats = defaultdict(lambda: [0, 0], {int(k): v for k, v in d["node_stats"].items()})
        s.code_stats = defaultdict(lambda: [0, 0], d["code_stats"])
        s.answer_log = d["answer_log"]
        s.trust = defaultdict(lambda: 1.0, d["trust"])
        s.review_log = d["review_log"]
        return s
