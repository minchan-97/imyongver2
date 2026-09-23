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

ap = argparse.ArgumentParser()
ap.add_argument("--subject", default=os.environ.get("SUBJECT", "국어"))
ap.add_argument("--fix", action="store_true")
a = ap.parse_args()

if not cloud.enabled():
    print("⚠️  SUPABASE_URL/KEY 없음 — 로컬 파일만 사용합니다.")
else:
    rep = cloud.sync(paths.all_paths(a.subject))
    print(f"동기화: 내려받음 {len(rep['down'])} · 올림 {len(rep['up'])}")

r = selfcheck.run(a.subject, fix=a.fix)
print(json.dumps({k: v for k, v in r.items() if k != "hygiene"},
                 ensure_ascii=False, indent=2, default=str))
print("경고:", r["alerts"] or "없음")
if cloud.ERRORS:
    print("서버 오류:", cloud.ERRORS)
