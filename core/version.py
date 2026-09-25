"""
version.py — 버전과 '이 버전이 필요로 하는 함수' 목록.

버전 줄을 모든 파일에 박아두면 한 줄만 고쳐도 전부 다시 올려야 해서,
여기 한 곳에만 둔다. 앱은 시작할 때 아래 목록으로 core 파일이
옛 버전인지 확인한다 (기능이 있는지로 확인하므로 파일 수정 여부와 무관).

새 기능을 만들면 그 함수 이름을 여기에 추가할 것.
"""
VERSION = "15.3"

REQUIRES = {
    "cloud": ["sync", "push", "list_uploads", "download_upload", "list_local_names",
              "archive_upload", "list_history", "restore_history"],
    "paths": ["all_paths", "discover_subjects", "scan_cache_path", "drive_state_path"],
    "schema": ["Record", "load_records_pkl", "save_records_pkl", "DOC_TYPES"],
    "page_scan": ["build_pages", "scan_pages", "looks_broken", "render_source"],
    "auto_tag": ["classify"],
    "drive": ["list_folder", "download", "DriveState", "dedupe", "folder_id"],
    "resubject": ["audit", "tag_gaps", "propagate_tags", "undetermined_docs",
                  "judge_docs_llm", "apply_doc_decisions", "judge",
                  "exam_pages", "judge_exam_pages", "apply_exam_moves"],
    "maintenance": ["run", "STEPS", "APP_STEPS", "is_garbage", "subject_list",
                    "clean_areas", "drop_garbage"],
    "selfcheck": ["run", "retrain", "auto_epochs", "load_history", "train_corpus"],
    "daily_digest": ["build", "mark_read", "recent", "today_str", "load_store"],
    "trend_lab": ["stats", "predict", "backtest", "summary", "load_lab", "top_rules"],
    "labeler": ["label_records", "train_tagger", "tag", "clean_area"],
    "ingest_queue": ["add", "pending", "run_queue", "load",
                     "enqueue_all_uploads", "enqueue_drive_new"],
    "self_exam": ["run", "load", "arm_table", "top_gaps", "retrieve", "retest_gaps"],
    "gap_search": ["collect", "pending", "accept", "reject", "domain_table",
                   "mark_helped"],
    "ask_box": ["generate", "pending", "answer", "skip", "summary", "mark_helped"],
    "inventory": ["subjects", "sources", "files", "queue", "summary"],
    "seed_rules": ["classify", "audit", "apply", "split_questions",
                   "split_exam_bucket"],
}


def check():
    """반환: 옛 버전으로 보이는 파일 설명 목록 (비어 있으면 정상)."""
    out = []
    for mod, attrs in REQUIRES.items():
        try:
            m = __import__(mod)
            missing = [a for a in attrs if not hasattr(m, a)]
            if missing:
                out.append(f"core/{mod}.py — 없는 기능: {', '.join(missing)}")
        except Exception as e:
            out.append(f"core/{mod}.py — 불러오기 실패: {e}")
    return out
