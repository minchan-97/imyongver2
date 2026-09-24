"""
trend_lab.py — 기출 경향 '예측 → 채점 → 규칙 누적' 원장.

설계 원칙
  1. 숫자는 LLM이 세지 않는다. 빈도·추세는 여기서 결정적으로 계산해
     '검증된 수치'로만 넘긴다(환각 차단).
  2. 예측은 기계가 채점할 수 있는 형태로만 받는다.
     {scope: area|code|keyword, target: 실제 존재하는 값, prob: 0~1}
  3. 모든 예측은 '언제, 무엇만 보고' 만들었는지와 함께 저장한다.
     → 과거 연도를 가려놓고 예측시키는 백테스트로 지금 바로 성적을 잴 수 있다.
  4. 예측마다 '규칙 한 문장'을 함께 받아 규칙별 성적을 누적한다.
     성적 좋은 규칙만 다음 예측의 few-shot으로 들어간다 = 데이터로 진화.

저장: data/trendlab_{과목}.pkl (서버 백업) — predictions / rules / runs
"""
from __future__ import annotations
CORE_VERSION = "13.3"
import os, re, json, time, pickle, hashlib
from collections import Counter, defaultdict

import paths
from korean_tokenizer import tokenize

try:
    from cloud import push as _cloud_push
except Exception:
    def _cloud_push(path, **kw): return False

SCOPES = ("area", "code", "keyword")
STOP = set("문항 다음 물음 답하시오 서술 설명 학생 교사 지도 내용 위해 대해 경우 것이 하는 이다".split())


def lab_path(subject):
    return paths._p(f"trendlab_{subject}.pkl")


def load_lab(subject):
    p = lab_path(subject)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"predictions": [], "rules": {}, "runs": []}


def save_lab(subject, lab):
    p = lab_path(subject)
    with open(p, "wb") as f:
        pickle.dump(lab, f)
    _cloud_push(p)


# ── 결정적 통계 ──────────────────────────────────────────────
def year_facts(l1, year, top_kw=25):
    """한 해 기출에서 실제로 나온 것들 (채점 기준이자 입력 통계)."""
    rs = [r for r in l1 if r.year == year]
    kw = Counter()
    for r in rs:
        for t in set(tokenize(r.text)):
            if len(t) >= 2 and t not in STOP:
                kw[t] += 1
    return {"year": year, "pages": len(rs),
            "areas": Counter(r.area for r in rs if r.area),
            "codes": Counter(r.code for r in rs if r.code),
            "keywords": Counter(dict(kw.most_common(top_kw)))}


def stats(l1, cutoff_year=None):
    """cutoff_year 이전(미포함) 자료만으로 만든 통계 — 백테스트 누수 방지."""
    use = [r for r in l1 if r.year and (cutoff_year is None or r.year < cutoff_year)]
    years = sorted({r.year for r in use})
    per = {y: year_facts(use, y) for y in years}
    recent = years[-3:]
    old = years[:-3]

    def agg(ys, key):
        c = Counter()
        for y in ys:
            c.update(per[y][key])
        return c
    out = {"years": years, "cutoff": cutoff_year, "pages": len(use), "per_year": per}
    for key in ("areas", "codes", "keywords"):
        r_, o_ = agg(recent, key), agg(old, key)
        out[key] = {"recent": dict(r_.most_common(20)), "past": dict(o_.most_common(20)),
                    "new": [k for k in r_ if k not in o_][:20],
                    "gone": [k for k in o_ if k not in r_][:20]}
    return out


def universe(st_):
    """예측 대상으로 허용할 값들 (기계 채점 가능한 것만)."""
    u = {}
    for key in ("areas", "codes", "keywords"):
        u[key[:-1] if key != "keywords" else "keyword"] = sorted(
            set(st_[key]["recent"]) | set(st_[key]["past"]))
    u["area"] = u.pop("area", u.get("area", []))
    return {"area": u.get("area", []), "code": u.get("code", []),
            "keyword": u.get("keyword", [])}


# ── 예측 ─────────────────────────────────────────────────────
SYSTEM = """너는 임용 기출 경향 분석가다. 아래 '검증된 통계'만 근거로 다음 시험을 예측한다.
통계에 없는 숫자를 지어내지 말 것. 아래 JSON 하나만 출력한다.
{"predictions":[
  {"scope":"area|code|keyword",
   "target":"허용 목록에 있는 값 그대로",
   "prob":0.0~1.0,          // 그 대상이 대상연도 기출에 '출제될' 확률
   "rule":"이 예측이 따르는 규칙 한 문장 (예: '3년 연속 출제된 영역은 이듬해에도 나온다')",
   "rationale":"통계 수치를 인용한 근거 한 줄"}
]}
규칙 10~20개. 확률은 정직하게. 확실하지 않으면 0.5 근처를 쓴다.
target은 반드시 허용 목록의 값과 글자까지 똑같아야 한다(채점이 자동이라 다르면 버려진다)."""


def _rule_id(text):
    return hashlib.sha1(re.sub(r"\s+", "", text or "").encode()).hexdigest()[:10]


def top_rules(lab, n=8, min_n=3):
    """성적 좋은 규칙 (적중률 높고 표본 있는 것) — 다음 예측의 few-shot."""
    out = []
    for rid, r in lab["rules"].items():
        if r["n"] >= min_n:
            out.append({"rule": r["text"], "n": r["n"],
                        "hit_rate": r["hits"] / r["n"],
                        "brier": r["brier_sum"] / r["n"]})
    out.sort(key=lambda x: (x["brier"], -x["hit_rate"]))
    return out[:n]


def predict(subject, l1, target_year, api_key, model="gpt-4o-mini", lab=None,
            cutoff=None):
    """cutoff(기본 target_year) 이전 자료만 보고 target_year를 예측."""
    cutoff = cutoff or target_year
    st_ = stats(l1, cutoff_year=cutoff)
    if not st_["years"]:
        raise ValueError("연도가 있는 기출이 없어요.")
    uni = universe(st_)
    lab = lab or load_lab(subject)
    good = top_rules(lab)
    brief = {
        "대상연도": target_year,
        "사용한_자료_연도": st_["years"],
        "연도별_쪽수": {y: st_["per_year"][y]["pages"] for y in st_["years"]},
        "영역": st_["areas"], "성취기준코드": st_["codes"], "핵심어": st_["keywords"],
        "허용_목록": uni,
        "지금까지_성적_좋았던_규칙": good,
    }
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(
        model=model, temperature=0.2, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": json.dumps(brief, ensure_ascii=False,
                                                         default=str)}])
    raw = json.loads(r.choices[0].message.content).get("predictions", [])
    preds, dropped = [], 0
    for p in raw:
        sc, tg = p.get("scope"), str(p.get("target", "")).strip()
        if sc not in SCOPES or tg not in uni.get(sc, []):
            dropped += 1
            continue
        try:
            prob = min(1.0, max(0.0, float(p.get("prob", 0.5))))
        except Exception:
            prob = 0.5
        rule = str(p.get("rule") or "").strip()
        preds.append({"id": hashlib.sha1(f"{subject}{target_year}{sc}{tg}{time.time()}"
                                         .encode()).hexdigest()[:12],
                      "subject": subject, "target_year": target_year, "cutoff": cutoff,
                      "scope": sc, "target": tg, "prob": prob,
                      "rule": rule, "rule_id": _rule_id(rule),
                      "rationale": str(p.get("rationale") or "")[:300],
                      "made_at": time.time(), "model": model,
                      "scored": False, "outcome": None, "brier": None})
    return preds, {"dropped": dropped, "stats": st_}


# ── 채점 ─────────────────────────────────────────────────────
def score(subject, l1, preds, lab):
    """대상연도 기출이 있는 예측만 채점하고 규칙 성적을 누적."""
    done = 0
    for p in preds:
        if p["scored"]:
            continue
        facts = year_facts(l1, p["target_year"])
        if facts["pages"] == 0:
            continue                      # 아직 그 해 기출이 없음 → 보류
        key = {"area": "areas", "code": "codes", "keyword": "keywords"}[p["scope"]]
        hit = p["target"] in facts[key]
        p["scored"], p["outcome"] = True, bool(hit)
        p["brier"] = (p["prob"] - (1.0 if hit else 0.0)) ** 2
        r = lab["rules"].setdefault(p["rule_id"], {"text": p["rule"], "n": 0,
                                                   "hits": 0, "brier_sum": 0.0})
        r["n"] += 1
        r["hits"] += int(hit)
        r["brier_sum"] += p["brier"]
        done += 1
    return done


def summary(preds):
    s = [p for p in preds if p["scored"]]
    if not s:
        return {"scored": 0}
    base = sum(p["outcome"] for p in s) / len(s)     # 무지성 기준선(기저율)
    return {"scored": len(s), "hit_rate": base,
            "brier": sum(p["brier"] for p in s) / len(s),
            "brier_baseline": base * (1 - base),     # 항상 기저율로 찍었을 때
            "by_scope": {sc: round(sum(p["brier"] for p in s if p["scope"] == sc)
                                   / max(1, len([p for p in s if p["scope"] == sc])), 4)
                         for sc in SCOPES}}


# ── 백테스트 ─────────────────────────────────────────────────
def backtest(subject, l1, api_key, model="gpt-4o-mini", years=None, progress=None):
    """과거 연도를 가려놓고 예측 → 실제와 대조. 지금 있는 자료만으로 성적이 나온다."""
    lab = load_lab(subject)
    all_years = sorted({r.year for r in l1 if r.year})
    targets = years or [y for y in all_years if len([x for x in all_years if x < y]) >= 2]
    out = []
    for i, y in enumerate(targets):
        if progress:
            progress(i, len(targets), y)
        try:
            preds, meta = predict(subject, l1, y, api_key, model, lab, cutoff=y)
        except Exception as e:
            out.append({"year": y, "error": str(e)})
            continue
        score(subject, l1, preds, lab)
        lab["predictions"] += preds
        out.append({"year": y, "n": len(preds), **summary(preds)})
    lab["runs"].append({"kind": "backtest", "at": time.time(), "years": targets,
                        "result": out})
    save_lab(subject, lab)
    return out, lab

