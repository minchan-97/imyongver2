"""
레이어3 — 최신 트렌드/논문 검색 (국어)

역할:
  L1(기출 패턴) + L2(자료 분포)를 바탕으로,
  '아직 자료가 성긴 개념영역(high QE)'이나 '최근 몰리는 개념'을
  검색어로 뽑아 Brave로 최신 논문/정책 자료를 가져온다.
  가져온 자료는 그냥 넣지 않고 → L2 국어 도메인 가드레일로 필터링한 뒤
  통과한 것만 트렌드 자료로 저장한다(드리프트 방지).

주의:
  - 이 환경은 네트워크가 막혀 있어 실제 Brave 호출은 사용자가 키를 꽂아 돌린다.
  - 여기서는 (1) 검색어 생성 (2) 결과 필터링 인터페이스만 완성해 둔다.
"""
from __future__ import annotations
import sys, os, json, urllib.parse, urllib.request
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))
from embedding import FrozenEmbedding
from som import SOM
from korean_tokenizer import tokenize
from schema import Record


def make_queries(emb: FrozenEmbedding, som: SOM, subject="국어", topn=8):
    """
    검색어 생성:
      high-QE 노드(잘 못 담은 영역)의 대표 단어 + 과목명으로 쿼리를 만든다.
    각 노드의 가중치 벡터에 가장 가까운 실제 어휘를 대표어로 뽑는다.
    """
    queries = []
    for node in som.high_qe_nodes(topn=topn):
        w = som.W[node]
        sims = emb.vectors @ w
        top_idx = sims.argsort()[::-1][:2]
        keywords = [emb.idx2word[i] for i in top_idx]
        q = f"{subject} 교육 {' '.join(keywords)} 최신 연구"
        queries.append({"node": node, "qe": float(som.qe[node]), "query": q})
    return queries


def brave_search(query, api_key, count=5):
    """
    Brave Search API 호출. (네트워크 되는 환경에서만 동작)
    사용자가 키를 꽂고 로컬에서 돌린다.
    """
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": query, "count": count})
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    results = []
    for item in data.get("web", {}).get("results", []):
        results.append({
            "title": item.get("title", ""),
            "text": item.get("description", ""),
            "source": item.get("url", ""),   # ← 출처 = URL, 스키마에 그대로 들어감
        })
    return results


def filter_by_guardrail(results, emb, som, subject="국어", pass_thr=0.40):
    """
    검색 결과를 L2 국어 도메인 가드레일로 필터.
    통과한 것만 L3_trend Record로 변환(출처 태그 유지).
    """
    kept = []
    for res in results:
        v = emb.embed_tokens(tokenize(res["text"]))
        if v is None:
            continue
        b = som.bmu_of(v)
        score = float(som.W[b] @ v)
        if score >= pass_thr:               # 국어 도메인 안이면 채택
            kept.append(Record(
                text=res["text"], layer="L3_trend", subject=subject,
                source=res["source"] or "brave:unknown",
            ))
    return kept


if __name__ == "__main__":
    emb = FrozenEmbedding.load("../embeddings/gukeo_emb.pkl")
    som = SOM.load("../embeddings/gukeo_som.pkl")
    qs = make_queries(emb, som)
    print("=== 생성된 검색어(‘잘 모르는 영역’ 기반) ===")
    print(json.dumps(qs, ensure_ascii=False, indent=2))
    print("\n실제 검색은 로컬에서:")
    print("  results = brave_search(q['query'], api_key='YOUR_KEY')")
    print("  kept = filter_by_guardrail(results, emb, som)")


# ══════════════════════════════════════════════════════════════
# 트렌드 속기사 + 사람 검토(반영/제거)
# ══════════════════════════════════════════════════════════════
def scribe_one_trend(result_text, api_key, model="gpt-4o-mini"):
    """검색 결과 하나를 '이게 어떤 트렌드인지' 한 줄 요약(속기)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    prompt = (
        "다음 검색 결과가 임용 국어 관점에서 '어떤 최신 흐름/개념'인지 "
        "한 문장으로 요약하라. 없는 내용 지어내지 말 것.\n\n"
        f"[검색결과]\n{result_text[:800]}\n\n출력: 한 문장 요약."
    )
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def prepare_trend_candidates(results, emb, som, subject, api_key=None, pass_thr=0.40):
    """
    검색 결과 → 가드레일 통과분에 속기 요약을 붙여 '후보'로 만든다.
    실제 저장은 사용자가 반영/제거를 고른 뒤(app에서).
    반환: [{text, source, score, summary, keep(기본True)}]
    """
    from korean_tokenizer import tokenize
    cands = []
    for res in results:
        v = emb.embed_tokens(tokenize(res["text"]))
        if v is None:
            continue
        b = som.bmu_of(v)
        score = float(som.W[b] @ v)
        if score < pass_thr:
            continue
        summary = ""
        if api_key:
            try:
                summary = scribe_one_trend(res["text"], api_key)
            except Exception:
                summary = "(요약 실패)"
        cands.append({"text": res["text"], "source": res["source"] or "brave:unknown",
                      "score": round(score, 3), "summary": summary, "keep": True})
    return cands
