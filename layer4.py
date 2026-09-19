"""
레이어4 — 연습문제 생성 + 출처 밝힌 해설 (국어)

역할:
  L1(패턴)+L2(자료)+L3(트렌드)를 종합해, 기출과 '유사한 개념 영역'의
  연습문제를 만든다. 생성은 LLM(속기사)이 하되, 판정·근거는 자체 자료에서.

핵심 원칙(합의됨):
  1) '예측기'가 아니라 '연습문제 생성기'. 맞히려는 게 아니라 그 개념을
     문제 형태로 여러 번 만나게 하는 인출연습(retrieval practice) 도구.
  2) 해설의 근거는 반드시 실제 자료의 '출처'를 밝혀서 가져온다.
     LLM이 지어낸 해설이 아니라 L2/L3 레코드의 출처가 붙는다.
  3) 해설은 사용자가 'reveal'(버튼)을 호출해야만 보인다.
     → 먼저 스스로 풀게 강제(학습효과의 핵심).

LLM 호출부는 인터페이스만. 사용자가 OpenAI 키 꽂아 로컬에서 돌린다.
"""
from __future__ import annotations
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))
from embedding import FrozenEmbedding
from som import SOM
from korean_tokenizer import tokenize


class Problem:
    """생성된 문제 하나. 해설은 reveal() 전엔 감춰진다."""
    def __init__(self, stem, concept_node, grounded_sources, model_answer=None):
        self.stem = stem                      # 발문(문제)
        self.concept_node = concept_node      # 어느 개념영역에서 나왔나
        self._grounded_sources = grounded_sources  # 근거 자료(출처 포함)
        self._model_answer = model_answer     # 해설(감춰둠)
        self._revealed = False

    def show(self):
        """문제만 보여준다(해설 감춤)."""
        return {"문제": self.stem, "개념영역": self.concept_node,
                "해설": "🔒 (풀어본 뒤 reveal() 하세요)"}

    def reveal(self):
        """버튼: 이걸 호출해야만 해설 + 출처가 보인다."""
        self._revealed = True
        return {
            "문제": self.stem,
            "해설": self._model_answer or "(LLM 미연결: 해설 생성 안 됨)",
            "근거_출처": self._grounded_sources,  # ← 출처 명시
        }


def pick_concept_nodes(som: SOM, l1_node_exams=None, topn=5):
    """
    문제를 낼 개념영역 선정:
      - 기출이 몰린 노드(빈출) 우선 (l1_node_exams 있으면)
      - 없으면 자료가 풍부한(레코드 많은) 노드
    """
    if l1_node_exams:
        nodes = sorted(l1_node_exams.keys(),
                       key=lambda n: -len(l1_node_exams[n]))
    else:
        nodes = sorted(range(som.gh * som.gw),
                       key=lambda n: -len(som.node_rec_ids.get(n, [])))
    return [n for n in nodes if len(som.node_rec_ids.get(n, [])) > 0][:topn]


def gather_grounding(node, som: SOM, records_by_id: dict, k=4):
    """해당 개념영역에 배정된 실제 자료 + 출처를 모은다(해설 근거)."""
    out = []
    for rid in som.node_rec_ids.get(node, [])[:k]:
        r = records_by_id.get(rid)
        if r:
            out.append({"내용": r.text[:80], "출처": r.source,
                        "성취기준코드": r.code, "연도": r.year})
    return out


# ── LLM 인터페이스(사용자가 키 꽂아 로컬 실행) ─────────────────────
def build_generation_prompt(grounding, exam_samples=None):
    """
    LLM에게 줄 프롬프트. '이 근거 자료 범위 안에서' 기출 유사 문제를 내라고 지시.
    범위를 자료로 묶어 환각을 억제한다.
    """
    grounds = "\n".join(f"- {g['내용']} (출처:{g['출처']})" for g in grounding)
    ex = ""
    if exam_samples:
        ex = "\n[참고 기출 형식]\n" + "\n".join(f"- {s}" for s in exam_samples[:3])
    return (
        "너는 초등 임용 국어 문제 출제 보조자다.\n"
        "아래 '근거 자료'의 개념 범위 안에서만, 기출과 유사한 서술형 연습문제 1개를 낸다.\n"
        "근거에 없는 내용을 지어내지 말 것. 해설은 근거 자료에 기반해 작성.\n\n"
        f"[근거 자료]\n{grounds}{ex}\n\n"
        "출력(JSON): {\"stem\": 문제, \"answer\": 해설}"
    )


def generate_with_llm(prompt, api_key, model="gpt-4o-mini"):
    """OpenAI 호출(로컬). 반환: (stem, answer)"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    txt = resp.choices[0].message.content
    try:
        d = json.loads(txt)
        return d.get("stem", txt), d.get("answer", "")
    except Exception:
        return txt, ""


def make_problems(som, emb, records_by_id, l1_node_exams=None,
                  api_key=None, exam_samples=None, n_nodes=5):
    """
    개념영역을 골라 문제를 만든다.
    api_key 없으면 근거·틀만 만들고 발문/해설은 비워둔다(구조 확인용).
    """
    problems = []
    for node in pick_concept_nodes(som, l1_node_exams, topn=n_nodes):
        grounding = gather_grounding(node, som, records_by_id)
        if api_key:
            prompt = build_generation_prompt(grounding, exam_samples)
            stem, answer = generate_with_llm(prompt, api_key)
        else:
            stem = f"[개념영역 {node}의 자료 기반 문제 — LLM 키 꽂으면 생성됨]"
            answer = None
        problems.append(Problem(stem, node, grounding, answer))
    return problems


if __name__ == "__main__":
    from schema import load_records
    emb = FrozenEmbedding.load("../embeddings/gukeo_emb.pkl")
    som = SOM.load("../embeddings/gukeo_som.pkl")
    recs = load_records("../data/gukeo_L2.json")
    by_id = {r.rec_id: r for r in recs if r.rec_id in
             {rid for ids in som.node_rec_ids.values() for rid in ids}}
    probs = make_problems(som, emb, by_id, api_key=None)
    print("=== 생성된 문제(해설 잠김) ===")
    for p in probs:
        print(json.dumps(p.show(), ensure_ascii=False, indent=2))
    if probs:
        print("\n=== reveal 예시(첫 문제 해설+출처) ===")
        print(json.dumps(probs[0].reveal(), ensure_ascii=False, indent=2))


# ══════════════════════════════════════════════════════════════
# 채점: 내 답 vs 모범답안 (LLM) + 정합도 보조신호
# ══════════════════════════════════════════════════════════════
def grade_answer_llm(question, my_answer, grounding, api_key, model="gpt-4o-mini"):
    """
    LLM으로 내 답을 채점 '제안'한다(최종 확정은 사용자).
    grounding(근거 자료)을 채점 기준으로 준다 → 근거 밖 잣대 억제.
    반환: dict(suggest: correct/partial/wrong, feedback, model_answer)
    """
    import json
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    grounds = "\n".join(f"- {g['내용']} (출처:{g['출처']})" for g in grounding)
    prompt = (
        "너는 초등 임용 국어 서술형 채점 보조자다.\n"
        "아래 '근거 자료'를 기준으로 학생 답안을 채점 '제안'하라.\n"
        "논리·핵심개념 포함 여부를 보고, 개념어만 겹친다고 정답 처리하지 말 것.\n\n"
        f"[문제]\n{question}\n\n[근거 자료]\n{grounds}\n\n[학생 답안]\n{my_answer}\n\n"
        "출력(JSON): {\"suggest\":\"correct|partial|wrong\","
        "\"feedback\":\"무엇이 맞고 무엇이 빠졌는지 2~3줄\","
        "\"model_answer\":\"근거에 기반한 모범답안\"}"
    )
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    txt = resp.choices[0].message.content
    try:
        d = json.loads(txt)
    except Exception:
        d = {"suggest": "partial", "feedback": txt, "model_answer": ""}
    return d


def grade_answer_offline(my_answer, emb, som, node):
    """
    LLM 없을 때: 내 답을 임베딩해 해당 개념영역(node)과의 유사도로
    '제안'만 한다. (개념어 겹침 수준 — 참고용임을 명시)
    """
    import numpy as np
    from korean_tokenizer import tokenize
    v = emb.embed_tokens(tokenize(my_answer))
    if v is None:
        return {"suggest": "wrong", "feedback": "개념 어휘가 거의 없음(참고용)",
                "model_answer": "", "score": 0.0}
    score = float(som.W[node] @ v)
    suggest = "correct" if score >= 0.55 else ("partial" if score >= 0.4 else "wrong")
    return {"suggest": suggest,
            "feedback": f"개념영역 유사도 {score:.2f} (※논리 채점 아님, 참고용)",
            "model_answer": "", "score": round(score, 3)}


def build_stem_only_prompt(grounding, exam_samples=None):
    """문제(stem)만 생성. 해설은 만들지 않는다(먼저 풀게 하려고)."""
    grounds = "\n".join(f"- {g['내용']} (출처:{g['출처']})" for g in grounding)
    ex = ""
    if exam_samples:
        ex = "\n[참고 기출 형식]\n" + "\n".join(f"- {s}" for s in exam_samples[:3])
    return (
        "너는 초등 임용 국어 문제 출제 보조자다.\n"
        "아래 '근거 자료'의 개념 범위 안에서만, 기출과 유사한 서술형 연습문제 1개를 낸다.\n"
        "근거에 없는 내용을 지어내지 말 것. **해설·정답은 절대 쓰지 말고 문제(발문)만** 출력.\n\n"
        f"[근거 자료]\n{grounds}{ex}\n\n"
        "출력(JSON): {\"stem\": 문제}"
    )


def generate_stem_only(prompt, api_key, model="gpt-4o-mini"):
    """문제만 생성해서 반환(str)."""
    import json
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    txt = resp.choices[0].message.content
    try:
        return json.loads(txt).get("stem", txt)
    except Exception:
        return txt


def build_answer_prompt(question, grounding):
    """reveal 시점에 해설·모범답안만 생성."""
    grounds = "\n".join(f"- {g['내용']} (출처:{g['출처']})" for g in grounding)
    return (
        "너는 초등 임용 국어 채점·해설 보조자다.\n"
        "아래 근거 자료에 기반해 이 문제의 모범답안과 해설을 작성하라.\n"
        "근거 밖 내용을 지어내지 말 것.\n\n"
        f"[문제]\n{question}\n\n[근거 자료]\n{grounds}\n\n"
        "출력(JSON): {\"model_answer\": 모범답안, \"explanation\": 해설}"
    )


def generate_answer(question, grounding, api_key, model="gpt-4o-mini"):
    """reveal 시점 해설 생성. 반환: (model_answer, explanation)"""
    import json
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_answer_prompt(question, grounding)}],
        temperature=0.2,
    )
    txt = resp.choices[0].message.content
    try:
        d = json.loads(txt)
        return d.get("model_answer", ""), d.get("explanation", "")
    except Exception:
        return txt, ""
