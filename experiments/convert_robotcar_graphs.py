import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "TGSeqLoc" / "src"))
from tgseqloc.data.robotcar_graphs import build_frame_index, convert_split

R = Path("/mnt/external_usb_hdd/6YL/Datasets/robotcar_llm")
OUT = Path.home() / "tgseqloc_data" / "robotcar_scenegraphs"
SPLITS = {
    "base": (R / "base_full_out" / "base_full.jsonl", R / "base_640_native"),
    "query": (R / "query_full_out" / "query_full.jsonl", R / "query_640_native"),
}
summary = {}
for role, (jsonl, chunks) in SPLITS.items():
    index = build_frame_index([chunks])
    print(f"{role}: индекс кадров {len(index)}")
    stats = convert_split(jsonl, index, OUT / role)
    summary[role] = stats.as_dict()
    print(f"  {json.dumps(stats.as_dict(), ensure_ascii=False)}")
    print(f"  топ классов: {', '.join(f'{c}({n})' for c, n in stats.classes.most_common(8))}")
(OUT / "conversion_stats.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
