"""
레이어1 — 기출 시계열 패턴 분석 (국어)

역할:
  역대 기출(초/중/특)을 연도 태그와 함께 L2의 고정 임베딩+SOM 위에 올려,
  '같은 개념 영역(BMU 클러스터)이 어느 해에 몰려 출제됐는가'를 본다.

패턴 정의(합의됨): '같은 개념 영역'.
  → 같은 SOM 노드(또는 인접 노드)에 배정된 기출들을 한 개념군으로 보고,
    그 개념군의 연도 분포를 확인한다.

급별 교차분석(합의됨):
  각 기출에 level(초/중/특) 태그가 있으므로,
  "중등에서 먼저 나온 개념이 몇 년 뒤 초등에 내려왔나"를 볼 수 있다.

주의: 이건 '예측기'가 아니라 '경향 분석기'다.
  어느 개념이 자주/최근 다뤄지는지를 보여줄 뿐, 올해 정답을 맞히지 않는다.
"""
from __future__ import annotations
import sys, os
import numpy as np
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))
from embedding import FrozenEmbedding
from som import SOM
from korean_tokenizer import tokenize
from schema import load_records


def map_exams_to_som(exam_records, emb: FrozenEmbedding, som: SOM):
    """기출 레코드를 (학습된) 국어 SOM에 얹어 BMU를 구한다. SOM은 재학습하지 않음."""
    node_exams = defaultdict(list)   # node -> [(year, level, rec)]
    for r in exam_records:
        v = emb.embed_tokens(tokenize(r.text))
        if v is None:
            continue
        b = som.bmu_of(v)
        node_exams[b].append((r.year, r.level, r))
    return node_exams


def concept_year_report(node_exams, min_hits=2):
    """
    각 개념영역(노드)별 연도 분포 리포트.
    min_hits: 이 개념군에 이만큼 이상 기출이 있어야 '패턴'으로 본다.
    """
    report = []
    for node, items in node_exams.items():
        if len(items) < min_hits:
            continue
        years = sorted(y for y, lv, r in items if y is not None)
        levels = [lv for y, lv, r in items if lv]
        sample = items[0][2].text[:50]
        report.append({
            "node": node,
            "hit_count": len(items),
            "years": years,
            "year_span": (min(years), max(years)) if years else None,
            "levels": dict(_count(levels)),
            "sample": sample,
        })
    report.sort(key=lambda d: -d["hit_count"])
    return report


def cross_level_flow(node_exams):
    """
    급별 교차분석: 같은 개념영역에서 급별로 '처음 등장한 해'를 비교.
    중등이 초등보다 먼저면 → 하향 전파 후보.
    """
    flows = []
    for node, items in node_exams.items():
        first_by_level = {}
        for y, lv, r in items:
            if y is None or not lv:
                continue
            if lv not in first_by_level or y < first_by_level[lv]:
                first_by_level[lv] = y
        if len(first_by_level) >= 2:  # 두 급 이상에서 등장
            flows.append({
                "node": node,
                "first_year_by_level": first_by_level,
                "sample": items[0][2].text[:50],
            })
    return flows


def _count(xs):
    d = defaultdict(int)
    for x in xs:
        d[x] += 1
    return d


if __name__ == "__main__":
    # python layer1.py ../data/gukeo_L1_exams.json
    exam_path = sys.argv[1] if len(sys.argv) > 1 else "../data/gukeo_L1_exams.json"
    emb = FrozenEmbedding.load("../embeddings/gukeo_emb.pkl")
    som = SOM.load("../embeddings/gukeo_som.pkl")
    exams = load_records(exam_path)
    node_exams = map_exams_to_som(exams, emb, som)

    import json
    print("=== 개념영역별 출제 연도 패턴 ===")
    print(json.dumps(concept_year_report(node_exams), ensure_ascii=False, indent=2))
    print("\n=== 급별 교차 흐름(초/중/특) ===")
    print(json.dumps(cross_level_flow(node_exams), ensure_ascii=False, indent=2))


# ══════════════════════════════════════════════════════════════
# 속기사: GasCore가 계산한 패턴 수치 → LLM이 '문장으로만' 옮김
# (판단은 SOM, LLM은 번역. 수치에 없는 경향을 추측하지 말 것)
# ══════════════════════════════════════════════════════════════
def build_pattern_facts(node_exams, som, emb, min_hits=2):
    """
    LLM에 넘길 '사실 수치'를 만든다. 여기엔 해석이 없다. 순수 데이터.
    """
    facts = []
    report = concept_year_report(node_exams, min_hits=min_hits)
    flows = cross_level_flow(node_exams)
    flow_by_node = {f["node"]: f for f in flows}
    for r in report:
        node = r["node"]
        # 이 개념영역의 대표 어휘(속기사가 개념을 지칭할 수 있게)
        w = som.W[node]
        top_idx = (emb.vectors @ w).argsort()[::-1][:3]
        keywords = [emb.idx2word[i] for i in top_idx]
        fact = {
            "개념영역_대표어": keywords,
            "출제_연도들": r["years"],
            "출제_횟수": r["hit_count"],
            "급별_분포": r["levels"],
            "예시문항": r["sample"],
        }
        if node in flow_by_node:
            fact["급별_최초등장"] = flow_by_node[node]["first_year_by_level"]
        facts.append(fact)
    return facts


def build_scribe_prompt(facts):
    """속기사 프롬프트 — 분석 금지, 수치 번역만."""
    import json
    return (
        "너는 임용 기출 경향 '속기사'다. 분석·추측하지 마라.\n"
        "아래는 시스템이 계산한 출제 패턴 수치다. 이 수치를 사람이 읽을\n"
        "자연스러운 한국어 경향 설명으로 '옮기기만' 하라.\n"
        "수치에 없는 주기·예측·경향을 지어내지 말 것. 있는 숫자만 서술.\n\n"
        f"[계산된 패턴 수치]\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
        "출력: 각 개념영역별로 2~3문장 경향 설명(간결하게)."
    )


def scribe_trends(node_exams, som, emb, api_key, min_hits=2, model="gpt-4o-mini"):
    """패턴 수치 → LLM 속기 → 경향 설명 텍스트."""
    facts = build_pattern_facts(node_exams, som, emb, min_hits=min_hits)
    if not facts:
        return "아직 반복 패턴이 없습니다(기출을 더 넣으면 개념영역별로 쌓입니다).", facts
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_scribe_prompt(facts)}],
        temperature=0.1,
    )
    return resp.choices[0].message.content, facts
