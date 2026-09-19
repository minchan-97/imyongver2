"""
레이어2 — 과목 자료 매핑 & 정합성 가드레일 (국어)

역할:
  성취기준/지도서/총론·각론 자료를 고정 임베딩 위 SOM에 올려,
  '국어라는 도메인의 정상 분포'를 만든다.
  이후 어떤 텍스트(문항/자료)를 넣으면:
    - 어느 개념 영역(BMU)에 속하는가
    - 그 영역에 얼마나 정합하는가(가드레일 판정)
  를 출처와 함께 돌려준다.

입력: L2_corpus 로 태그된 국어 Record 리스트.
출력: 학습된 SOM(embeddings/gukeo_som.pkl) + 판정 함수.
"""
from __future__ import annotations
import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))
from embedding import FrozenEmbedding, train_embedding
from som import SOM
from korean_tokenizer import tokenize
from schema import Record, load_records


def build_layer2(records, emb_path, som_path, dim=64, grid=(12, 12),
                 emb_epochs=5, som_iters=3000):
    """
    국어 자료로 (1) 임베딩 학습·고정 (2) SOM 학습·저장.
    records: L2_corpus 국어 Record 리스트.
    """
    texts = [r.text for r in records]

    # (1) 임베딩: 한 번 학습해서 고정
    if os.path.exists(emb_path):
        print(f"[L2] 기존 고정 임베딩 로드: {emb_path}")
        emb = FrozenEmbedding.load(emb_path)
    else:
        print("[L2] 임베딩 학습(최초 1회, 이후 재사용)...")
        emb = train_embedding(texts, dim=dim, epochs=emb_epochs)
        emb.save(emb_path)
        print(f"[L2] 임베딩 고정 저장: {emb_path}")

    # (2) 각 레코드 → 벡터 (OOV로 벡터 못 만든 레코드는 제외)
    X, kept = [], []
    for r in records:
        v = emb.embed_tokens(tokenize(r.text))
        if v is not None:
            X.append(v); kept.append(r)
    X = np.array(X)
    print(f"[L2] 벡터화된 레코드: {len(kept)}/{len(records)}")

    # (3) SOM 학습(국어만) + 배정
    som = SOM(grid=grid, dim=dim)
    som.train(X, iters=som_iters)
    som.assign(X, kept)
    som.save(som_path)
    print(f"[L2] SOM 저장: {som_path}")
    return emb, som, kept


def judge(text, emb: FrozenEmbedding, som: SOM, records_by_id: dict,
          pass_thr=0.45):
    """
    자료 정합성 가드레일.
    반환: dict(정합도, 판정, 가까운 개념영역의 출처들)
    pass_thr: BMU 코사인 유사도 임계값(도메인 안/밖). 튜닝 대상.
    """
    v = emb.embed_tokens(tokenize(text))
    if v is None:
        return {"verdict": "판정불가", "reason": "임베딩 불가(모르는 단어뿐)", "score": None}
    b = som.bmu_of(v)
    score = float(som.W[b] @ v)               # BMU 유사도
    verdict = "정합" if score >= pass_thr else "이탈(국어 도메인 밖 가능)"
    # 이 개념영역(노드)에 배정된 자료들의 출처 = 근거 후보
    src_ids = som.node_rec_ids.get(b, [])[:5]
    sources = []
    for rid in src_ids:
        r = records_by_id.get(rid)
        if r:
            sources.append({"text": r.text[:60], "source": r.source,
                            "code": r.code, "year": r.year})
    return {
        "verdict": verdict, "score": round(score, 3),
        "bmu": b, "concept_sources": sources,
    }


if __name__ == "__main__":
    # 사용 예:
    #   python layer2.py data/gukeo_L2.json
    import sys
    from schema import load_records
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../data/gukeo_L2.json"
    recs = load_records(data_path)
    emb, som, kept = build_layer2(
        recs,
        emb_path="../embeddings/gukeo_emb.pkl",
        som_path="../embeddings/gukeo_som.pkl",
    )
    by_id = {r.rec_id: r for r in kept}
    # 데모 판정
    test = "학생이 이야기를 읽고 인물의 마음을 짐작하여 표현한다"
    print("\n[판정 데모]", test)
    import json
    print(json.dumps(judge(test, emb, som, by_id), ensure_ascii=False, indent=2))
