"""Проверка конвертера на реальных данных.

regress — пересобрать графы первого поколения во временный каталог и сравнить
          побайтно с уже сконвертированными: правка не должна их менять;
new     — сконвертировать графы второго поколения в отдельный каталог.
"""
import filecmp, json, shutil, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path.home() / "TGSeqLoc" / "src"))
from tgseqloc.data.robotcar_graphs import build_frame_index, convert_split

R = Path("/mnt/external_usb_hdd/6YL/Datasets/robotcar_llm")
OLD_OUT = Path.home() / "tgseqloc_data" / "robotcar_scenegraphs"
NEW_OUT = Path.home() / "tgseqloc_data" / "robotcar_scenegraphs_gemini"
OLD = {"base": (R / "base_full_out/base_full.jsonl", R / "base_640_native"),
       "query": (R / "query_full_out/query_full.jsonl", R / "query_640_native")}
NEW = {"base": (R / "base_openrouter_final_out/gemini-3.1-flash-lite-base.jsonl",
                R / "base_openrouter_final/frames_640"),
       "query": (R / "query_openrouter_final_out/gemini-3.1-flash-lite-query.jsonl",
                 R / "query_openrouter_final/frames_640")}

if sys.argv[1] == "regress":
    tmp = Path(tempfile.mkdtemp(prefix="rc_regress_"))
    ok = True
    try:
        for role, (jsonl, chunks) in OLD.items():
            stats = convert_split(jsonl, build_frame_index([chunks]), tmp / role)
            before = sorted(p.name for p in (OLD_OUT / role).glob("*.json"))
            after = sorted(p.name for p in (tmp / role).glob("*.json"))
            match, mismatch, errors = filecmp.cmpfiles(OLD_OUT / role, tmp / role, before, shallow=False)
            same = before == after and not mismatch and not errors
            ok &= same
            print("%-5s файлов было %d, стало %d | побайтно совпало %d, отличается %d, ошибок %d | %s"
                  % (role, len(before), len(after), len(match), len(mismatch), len(errors),
                     "ИДЕНТИЧНО" if same else "РАСХОЖДЕНИЕ"))
            d = stats.as_dict()
            print("      потери: боксов %d, связей %d | вырожденных %d, без таблицы %d | раскладки %s"
                  % (d["dropped_boxes"], d["dropped_relations"], d["degenerate"], d["unparsed"],
                     d["object_layouts"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)

summary = {}
for role, (jsonl, chunks) in NEW.items():
    index = build_frame_index([chunks])
    stats = convert_split(jsonl, index, NEW_OUT / role)
    summary[role] = stats.as_dict()
    d = summary[role]
    print("%-5s индекс %d | записей %d, записано %d | без таблицы %d, вырожденных %d, пустых %d, кадр не найден %d"
          % (role, len(index), d["records"], d["written"], d["unparsed"], d["degenerate"],
             d["empty"], d["unknown_frame"]))
    print("      потери: боксов %d, связей %d | классов %d | раскладки %s"
          % (d["dropped_boxes"], d["dropped_relations"], d["distinct_classes"], d["object_layouts"]))
    print("      топ классов:", ", ".join("%s(%d)" % kv for kv in stats.classes.most_common(8)))
(NEW_OUT / "conversion_stats.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
