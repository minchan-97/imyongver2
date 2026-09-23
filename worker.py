"""
worker.py — 서버에서 혼자 도는 점검·재학습.

    SUPABASE_URL=... SUPABASE_KEY=... python scripts/worker.py --subject 국어 --fix

1) 서버에서 최신 pkl을 내려받고
2) 자기검증(데이터 위생·모델 품질·회귀 테스트)을 돌리고
3) --fix면 기준 미달일 때 임베딩·SOM을 재학습해서
4) 결과 pkl과 리포트를 다시 서버에 올린다.

GitHub Actions·크론·내 PC 어디서 돌려도 같다. 앱은 사이드바에서 그 리포트를 본다.
"""
import os, sys, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

import paths, cloud, selfcheck


def run_one(subject, fix):
    r = selfcheck.run(subject, fix=fix)
    # 새 기출이 들어왔으면 대기 중인 경향 예측을 자동 채점 (LLM 호출 없음)
    try:
        import trend_lab as tl
        from schema import load_records_pkl
        lab = tl.load_lab(subject)
        n = tl.score(subject, load_records_pkl(paths.l1_path(subject)),
                     lab["predictions"], lab)
        if n:
            tl.save_lab(subject, lab)
        print(f"경향 예측 채점: {n}건 · 누적 {tl.summary(lab['predictions'])}")
    except Exception as e:
        print("경향 채점 건너뜀:", e)
    print(json.dumps({k: v for k, v in r.items() if k != "hygiene"},
                     ensure_ascii=False, indent=2, default=str))
    print("경고:", r["alerts"] or "없음")
    return r

ap = argparse.ArgumentParser()
ap.add_argument("--subject", default=os.environ.get("SUBJECT", "all"),
                help="과목명, 쉼표로 여러 개, 또는 'all'(자료가 있는 과목 전부)")
ap.add_argument("--fix", action="store_true")
a = ap.parse_args()

if not cloud.enabled():
    print("⚠️  SUPABASE_URL/KEY 없음 — 로컬 파일만 사용합니다.")

if a.subject.strip().lower() in ("all", "*", "전체"):
    subjects = paths.discover_subjects(cloud.list_local_names())
    if not subjects:
        print("자료가 있는 과목을 못 찾았어요. --subject 국어 처럼 직접 지정하세요.")
        sys.exit(0)
else:
    subjects = [s.strip() for s in a.subject.split(",") if s.strip()]
print("대상 과목:", ", ".join(subjects))

summary = []
for subject in subjects:
    print(f"\n════ {subject} ════")
    if cloud.enabled():
        rep = cloud.sync(paths.all_paths(subject))
        print(f"동기화: 내려받음 {len(rep['down'])} · 올림 {len(rep['up'])}")
    try:
        r = run_one(subject, a.fix)
        summary.append((subject, len(r["alerts"]), bool(r.get("retrained"))))
    except Exception as e:
        print(f"⚠️  {subject} 실패: {e}")
        summary.append((subject, -1, False))

print("\n════ 요약 ════")
for s, n, rt in summary:
    print(f"{s}: " + ("실패" if n < 0 else f"경고 {n}건") + (" · 재학습함" if rt else ""))
if cloud.ERRORS:
    print("서버 오류:", cloud.ERRORS)

