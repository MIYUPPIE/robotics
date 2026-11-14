"""Periodic eval (paid, real Claude). Text-level Yoruba->English quality.
Run: python evals/eval_translate.py   Threshold: >= 80% keyword hit rate.
Audio eval: drop evals/clips/<name>.wav + <name>.txt (reference English) and
they are run through the full pipeline, reporting WER-free keyword hit rate."""
import json, pathlib, sys, unicodedata
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline, translate as tr  # noqa: E402

THRESHOLD = 0.8
# (yoruba, keywords any of which must appear, lowercase)
CASES = [
    ("Ẹ jọ̀ọ́, ṣé ẹ lè ràn mí lọ́wọ́?", [["help"], ["please"]]),
    ("Ẹ káàárọ̀", [["good morning", "morning"]]),
    ("Orúkọ mi ni Adé", [["name"], ["ade"]]),
    ("Mo fẹ́ omi", [["water"], ["want", "would like"]]),
    ("Ṣé o ti jẹun?", [["eat", "eaten", "food", "meal"]]),
]


def fold(t):
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if not unicodedata.combining(c))


def score(en, groups):
    en = fold(en)
    return all(any(k in en for k in g) for g in groups)


def main():
    rows = []
    for yo, groups in CASES:
        en = tr.translate(yo)
        rows.append({"yoruba": yo, "english": en, "pass": score(en, groups)})
    for clip in sorted((ROOT / "evals" / "clips").glob("*.wav")):
        ref = clip.with_suffix(".txt")
        if ref.exists():
            s = pipeline.run(str(clip))
            kws = [[w] for w in ref.read_text().lower().split() if len(w) > 3]
            rows.append({"clip": clip.name, "yoruba": s.yoruba, "english": s.english,
                         "pass": sum(any(k in s.english.lower() for k in g) for g in kws) / max(len(kws), 1) >= 0.5})
    rate = sum(r["pass"] for r in rows) / len(rows)
    (ROOT / "evals" / "results.json").write_text(json.dumps({"rate": rate, "rows": rows}, ensure_ascii=False, indent=2))
    print(json.dumps(rows, ensure_ascii=False, indent=2)); print(f"pass rate {rate:.0%} (threshold {THRESHOLD:.0%})")
    sys.exit(0 if rate >= THRESHOLD else 1)


if __name__ == "__main__":
    main()
