"""
concept_dict.py — 과목별 '개념 태그 사전'.

각론을 읽으며 내가 만든 개념 태그를 목록으로 쌓는다.
표기 일관성이 생명('사건의 흐름' vs '사건 흐름'이 갈라지면 연결이 끊김).
그래서 한 번 만든 개념은 사전에서 골라 재사용한다.
"""
from __future__ import annotations
CORE_VERSION = "13.3"
import os, pickle

try:
    from cloud import push as _cloud_push
except Exception:  # cloud 계층이 없어도 로컬 동작은 유지
    def _cloud_push(path, **kw): return False



class ConceptDict:
    def __init__(self, subject):
        self.subject = subject
        self.concepts = []   # [{"name":..., "area":..., "note":...}]

    def add(self, name, area=None, note=""):
        name = name.strip()
        if not name:
            return False
        if any(c["name"] == name for c in self.concepts):
            return False   # 중복
        self.concepts.append({"name": name, "area": area, "note": note})
        return True

    def names(self):
        return [c["name"] for c in self.concepts]

    def by_area(self, area):
        return [c for c in self.concepts if c.get("area") == area]

    def remove(self, name):
        self.concepts = [c for c in self.concepts if c["name"] != name]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"subject": self.subject, "concepts": self.concepts}, f)
        _cloud_push(path)

    @staticmethod
    def load(path, subject):
        d = ConceptDict(subject)
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            d.concepts = data.get("concepts", [])
        return d
