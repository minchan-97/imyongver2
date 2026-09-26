"""
self_exam.py — 워커가 밤에 스스로 문제를 내고 풀고 채점한다.

가장 중요한 설계: **채점을 LLM에게 맡기지 않는다.**
LLM이 내고 LLM이 풀고 LLM이 맞다고 하면 셋 다 같은 방향으로 틀린다.
그래서 정답이 기계적으로 확인되는 두 가지만 낸다.

  A) 근거 검색 시험 (retrieval)  — LLM 없이, 공짜
     한 기록을 질문으로 삼아 "같은 성취기준/같은 문서의 다른 쪽"을 찾아오게 한다.
     정답은 corpus가 갖고 있다 → 맞았는지 즉시 확인 가능.

  B) 빈칸 복원 시험 (cloze)      — LLM 1회
     기록에서 핵심 낱말을 가리고, **다른 기록들만** 근거로 주고 복원하게 한다.
     정답은 가린 그 낱말 → 문자열 비교로 채점. 모델의 자기 평가가 끼지 않는다.

배우는 것: '근거를 어떻게 찾을까'의 전략 선택.
  code(성취기준 코드) / node(SOM 개념영역) / keyword(낱말 겹침) / embed(임베딩 유사도)
전략마다 성적을 쌓고 UCB로 고른다(많이 쓴 것과 덜 써본 것의 균형).
가중치를 고치는 게 아니므로 엄밀한 강화학습은 아니고 bandit에 가깝다 — 이 규모엔 이게 정직하다.

결핍 기록: 어떤 성취기준·개념에서 계속 근거를 못 찾는지 gaps에 쌓인다.
→ gap_search.py가 그 구멍만 웹에서 찾는다.

저장: data/selfexam_{과목}.pkl
"""
from __future__ import annotations
import os, re, math, json, time, pickle, random
from collections import Counter, defaultdict

import numpy as np
import paths
from schema import load_records_pkl
from korean_tokenizer import tokenize

try:
    from cloud import push as _cloud_push
except Exception:
    def _cloud_push(path, **kw): return False

ARMS = ["code", "node", "keyword", "embed"]
STOP = set("국어 학생 교사 지도 내용 활동 수업 학습 방법 경우 위해 대해 자료 단원 "
           "차시 교육 평가 성취 기준 있다 하는 것이 NUM".split())


def lab_path(subject):
    return paths._p(f"selfexam_{subject}.pkl")


def load(subject):
    p = lab_path(subject)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"arms": {a: {"n": 0, "wins": 0.0} for a in ARMS},
            "runs": [], "gaps": {}, "history": []}


def save(subject, lab):
    lab["runs"] = lab["runs"][-50:]
    lab["history"] = lab["history"][-500:]
    p = lab_path(subject)
    with open(p, "wb") as f:
        pickle.dump(lab, f)
    _cloud_push(p)


# ── 전략(팔) 고르기 ──────────────────────────────────────────
def pick_arm(lab, rng=random):
    """UCB1 — 성적 좋은 전략을 더 쓰되, 덜 써본 전략도 가끔 시험한다."""
    arms = lab["arms"]
    total = sum(a["n"] for a in arms.values()) + 1
    best, best_v = None, -1e9
    for name in ARMS:
        a = arms.setdefault(name, {"n": 0, "wins": 0.0})
        if a["n"] == 0:
            return name
        v = a["wins"] / a["n"] + math.sqrt(2 * math.log(total) / a["n"])
        if v > best_v:
            best, best_v = name, v
    return best


def arm_table(lab):
    out = []
    for name in ARMS:
        a = lab["arms"].get(name, {"n": 0, "wins": 0.0})
        out.append({"전략": name, "시행": a["n"],
                    "성공률": (a["wins"] / a["n"]) if a["n"] else None})
    out.sort(key=lambda x: -(x["성공률"] or -1))
    return out


# ── 근거 찾기 전략들 ─────────────────────────────────────────
def _kw(text, n=12):
    c = Counter(t for t in tokenize(text) if len(t) >= 2 and t not in STOP)
    return [w for w, _ in c.most_common(n)]


def retrieve(arm, query_rec, corpus, emb=None, som=None, k=5):
    """query_rec를 제외하고 근거 후보 k개를 찾아온다."""
    pool = [r for r in corpus if r.rec_id != query_rec.rec_id]
    if not pool:
        return []
    if arm == "code" and query_rec.code:
        hit = [r for r in pool if r.code == query_rec.code]
        if hit:
            return hit[:k]
        return []
    if arm == "node" and som is not None and emb is not None:
        v = emb.embed_tokens(tokenize(query_rec.text))
        if v is None:
            return []
        node = som.bmu_of(v)
        ids = set(som.node_rec_ids.get(node, []))
        hit = [r for r in pool if r.rec_id in ids]
        return hit[:k]
    if arm == "embed" and emb is not None:
        v = emb.embed_tokens(tokenize(query_rec.text))
        if v is None:
            return []
        scored = []
        for r in pool:
            w = emb.embed_tokens(tokenize(r.text))
            if w is not None:
                scored.append((float(v @ w), r))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:k]]
    # keyword (기본)
    want = set(_kw(query_rec.text))
    scored = [(len(want & set(_kw(r.text))), r) for r in pool]
    scored.sort(key=lambda x: -x[0])
    return [r for s, r in scored[:k] if s > 0]


# ── A) 근거 검색 시험 (LLM 없음) ─────────────────────────────
def _same_topic(a, b):
    """정답 판정: 같은 성취기준 코드 또는 같은 문서(출처)면 관련 있다고 본다."""
    if a.code and b.code:
        return a.code == b.code
    base = lambda r: r.source.rsplit(" (", 1)[0].rsplit(" p.", 1)[0]
    return base(a) == base(b)


def run_retrieval(subject, corpus, lab, n=30, emb=None, som=None, rng=None, log=print):
    rng = rng or random.Random(0)
    # 같은 코드/문서의 다른 쪽이 실제로 있는 기록만 문제로 낸다 (정답이 존재해야 채점 가능)
    base = lambda r: r.source.rsplit(" (", 1)[0].rsplit(" p.", 1)[0]
    by_key = defaultdict(list)
    for r in corpus:
        by_key[r.code or base(r)].append(r)
    candidates = [r for r in corpus if len(by_key[r.code or base(r)]) >= 2]
    if not candidates:
        return {"n": 0, "note": "정답을 확인할 수 있는 문제를 못 만들었어요(자료가 적음)"}
    rng.shuffle(candidates)
    wins, tried = 0, 0
    for q in candidates[:n]:
        arm = pick_arm(lab, rng)
        got = retrieve(arm, q, corpus, emb, som)
        ok = any(_same_topic(q, r) for r in got)
        a = lab["arms"][arm]
        a["n"] += 1
        a["wins"] += 1.0 if ok else 0.0
        tried += 1
        wins += int(ok)
        key = q.code or base(q)
        lab["history"].append({"kind": "retrieval", "arm": arm, "ok": ok,
                               "key": key, "at": time.time()})
        if ok and key in lab["gaps"] and not lab["gaps"][key].get("fixed"):
            lab["gaps"][key]["fixed"] = True        # 구멍이 메워짐
            _credit(subject, key)                    # 답·웹자료에 공을 돌린다
        if not ok:
            _mark_gap(lab, q, "근거 검색 실패")
    return {"n": tried, "ok": wins, "rate": wins / max(tried, 1)}


# ── B) 빈칸 복원 시험 (LLM 1회) ──────────────────────────────
CLOZE_SYSTEM = """너는 주어진 '근거 자료'만 보고 빈칸에 들어갈 낱말을 맞히는 응시자다.
근거에 없으면 추측하지 말고 "모름"이라고 답한다. JSON 하나만 출력:
{"answer":"낱말만", "quote":"근거에서 그대로 옮긴 한 문장"}"""


def _norm(s):
    return re.sub(r"[^가-힣A-Za-z0-9]", "", str(s or "")).lower()


def run_cloze(subject, corpus, lab, api_key, model="gpt-4o-mini", n=10,
              emb=None, som=None, rng=None, log=print):
    if not api_key:
        return {"n": 0, "note": "OpenAI 키가 없어 건너뜀"}
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    rng = rng or random.Random(0)
    pool = [r for r in corpus if len(r.text) > 120]
    rng.shuffle(pool)
    wins, tried, items = 0, 0, []
    for q in pool:
        if tried >= n:
            break
        terms = [t for t in _kw(q.text, 6) if len(t) >= 2]
        if not terms:
            continue
        term = terms[0]
        masked = q.text.replace(term, "____")
        if "____" not in masked:
            continue
        arm = pick_arm(lab, rng)
        ev = retrieve(arm, q, corpus, emb, som, k=4)
        if not ev:
            _mark_gap(lab, q, "근거 없음")
            lab["arms"][arm]["n"] += 1
            tried += 1
            items.append({"term": term, "arm": arm, "ok": False, "why": "근거 못 찾음"})
            continue
        body = "\n\n".join(f"[{r.source}]\n{r.text[:800]}" for r in ev)
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": CLOZE_SYSTEM},
                          {"role": "user",
                           "content": f"근거 자료:\n{body}\n\n문제(빈칸 채우기):\n{masked[:1500]}"}])
            d = json.loads(resp.choices[0].message.content)
            ans = str(d.get("answer", ""))
            quote = str(d.get("quote", ""))
        except Exception as e:
            msg = str(e)
            if "rate_limit" in msg or "429" in msg:
                log("    속도 제한 — 빈칸 시험을 여기서 멈춥니다")
                break                      # 같은 오류로 로그를 도배하지 않는다
            log(f"    cloze 실패: {msg[:120]}")
            continue
        ok = _norm(ans) == _norm(term) or (_norm(term) in _norm(ans) and len(term) >= 2)
        # 근거 인용이 실제 자료에 있는지도 함께 본다(지어낸 인용 걸러내기)
        quoted_real = any(_norm(quote)[:20] and _norm(quote)[:20] in _norm(r.text)
                          for r in ev) if quote else False
        a = lab["arms"][arm]
        a["n"] += 1
        a["wins"] += 1.0 if ok else 0.0
        tried += 1
        wins += int(ok)
        items.append({"term": term, "answer": ans, "arm": arm, "ok": ok,
                      "quote_real": quoted_real, "source": q.source})
        lab["history"].append({"kind": "cloze", "arm": arm, "ok": ok,
                               "quote_real": quoted_real, "at": time.time()})
        if not ok:
            _mark_gap(lab, q, f"빈칸 복원 실패({term})")
    return {"n": tried, "ok": wins, "rate": wins / max(tried, 1), "items": items[:10]}


def _credit(subject, gap_key):
    """구멍이 메워졌을 때, 그 구멍에 기여한 사람 답변·웹 출처의 성적을 올린다."""
    for mod in ("ask_box", "gap_search"):
        try:
            m = __import__(mod)
            m.mark_helped(subject, gap_key, True)
        except Exception:
            pass


# ── 결핍(구멍) 기록 ─────────────────────────────────────────
def _mark_gap(lab, rec, why):
    key = rec.code or (rec.area or "") or rec.source.rsplit(" p.", 1)[0]
    g = lab["gaps"].setdefault(key, {"key": key, "misses": 0, "why": why,
                                     "area": rec.area, "code": rec.code,
                                     "sample": rec.text[:120], "subject": rec.subject,
                                     "searched": False, "fixed": False})
    g["misses"] += 1
    g["why"] = why
    g["last"] = time.time()


def prune_gaps(lab, keep=200):
    """메워진 지 오래된 구멍은 정리한다 (목록이 끝없이 커지지 않게)."""
    gaps = lab.get("gaps") or {}
    if len(gaps) <= keep:
        return 0
    items = sorted(gaps.items(),
                   key=lambda kv: (not kv[1].get("fixed"), kv[1].get("last", 0)))
    drop = [k for k, _ in items[:len(gaps) - keep]]
    for k in drop:
        gaps.pop(k, None)
    return len(drop)


def top_gaps(lab, n=10, only_unsearched=False):
    gs = [g for g in lab["gaps"].values()
          if not g.get("fixed") and (not only_unsearched or not g.get("searched"))]
    gs.sort(key=lambda g: -g["misses"])
    return gs[:n]


# ── 답한 구멍 다시 풀어보기 ─────────────────────────────────
def _fills_gap(rec, key):
    """이 기록이 그 구멍을 겨냥한 것인가 (코드·영역 일치 또는 답변/웹수집의 출처 표기)."""
    if rec.code and rec.code == key:
        return True
    if rec.area and rec.area == key:
        return True
    if rec.doc_type in ("내_답변", "웹수집") and key and key in (rec.source or ""):
        return True
    if rec.doc_type in ("내_답변", "웹수집") and key and key in (rec.text or "")[:200]:
        return True
    return False


def retest_gaps(subject, corpus, lab, emb=None, som=None, log=print):
    """
    답변·웹수집으로 메워졌을 법한 구멍만 골라 다시 풀어본다.
    (무작위 시험에 그 구멍이 다시 뽑히기를 기다리지 않고 직접 확인한다)
    판정: 구멍의 본문을 질문으로 삼아 근거를 찾았을 때,
          그 구멍을 겨냥한 자료가 실제로 검색되면 '메움'.
    """
    from schema import Record
    fixed, checked = [], 0
    for key, g in list(lab["gaps"].items()):
        if g.get("fixed"):
            continue
        # 열린 구멍은 모두 다시 풀어본다. 자료가 늘어서 저절로 메워진 것도 닫아야
        # 구멍 목록이 무한정 쌓이지 않는다.
        # 다만 '누구 덕분인지'(credit)는 내 답변·웹수집이 있을 때만 준다.
        targets = [r for r in corpus if _fills_gap(r, key)]
        by_new = any(r.doc_type in ("내_답변", "웹수집") for r in targets)
        checked += 1
        sample = (g.get("sample") or key)
        probe = Record(text=sample if len(sample) > 15 else f"{key} {sample}",
                       layer="L2_corpus", subject=subject, source="probe")
        found = False
        for arm in ARMS:
            got = retrieve(arm, probe, corpus, emb, som, k=5)
            if any(_fills_gap(r, key) for r in got):
                found = True
                break
        if found:
            g["fixed"] = True
            g["fixed_at"] = time.time()
            g["fixed_by"] = "새 자료" if by_new else "기존 자료"
            fixed.append(key)
            if by_new:
                _credit(subject, key)
            log(f"    구멍 메움: {key} ({g['fixed_by']})")
    return {"checked": checked, "fixed": fixed}


# ── 한 회차 ─────────────────────────────────────────────────
def run(subject, api_key=None, model="gpt-4o-mini", n_retrieval=30, n_cloze=8,
        log=print):
    from embedding import FrozenEmbedding
    from som import SOM
    import selfcheck as _sc
    corpus = _sc.train_corpus(subject)      # L2 + L1(기출) + 공통
    if len(corpus) < 5:
        return {"error": f"{subject}: 자료가 적어 자가 시험을 건너뜁니다({len(corpus)}건)"}
    emb = (FrozenEmbedding.load(paths.emb_path(subject))
           if os.path.exists(paths.emb_path(subject)) else None)
    som = (SOM.load(paths.som_path(subject))
           if os.path.exists(paths.som_path(subject)) else None)
    lab = load(subject)
    rng = random.Random(int(time.time()) % 100000)

    r0 = retest_gaps(subject, corpus, lab, emb, som, log)
    if r0["fixed"]:
        log(f"  다시 풀어보니 메워진 구멍 {len(r0['fixed'])}곳")
    r1 = run_retrieval(subject, corpus, lab, n_retrieval, emb, som, rng, log)
    log(f"  근거 검색 시험: {r1.get('ok', 0)}/{r1.get('n', 0)}")
    r2 = run_cloze(subject, corpus, lab, api_key, model, n_cloze, emb, som, rng, log)
    log(f"  빈칸 복원 시험: {r2.get('ok', 0)}/{r2.get('n', 0)}")

    rep = {"subject": subject, "at": time.time(), "retest": r0,
           "retrieval": r1, "cloze": r2,
           "arms": arm_table(lab), "gaps": len(lab["gaps"])}
    rep["gaps_open"] = sum(1 for g in lab["gaps"].values() if not g.get("fixed"))
    rep["gaps_fixed"] = sum(1 for g in lab["gaps"].values() if g.get("fixed"))
    prune_gaps(lab)
    lab["runs"].append(rep)
    save(subject, lab)
    return rep
