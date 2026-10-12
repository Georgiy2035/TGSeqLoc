"""R@1 R@5 R@10 одной строкой, в точном представлении, — для побитного сравнения."""
import json, sys

try:
    text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    obj, _ = json.JSONDecoder().raw_decode(text[text.find("{"):])
except Exception:
    sys.exit(0)
m = obj.get("final_metrics") or obj.get("training", {}).get("final_metrics") or (obj if "R@5" in obj else None)
if m:
    print(" ".join(repr(float(m[k])) for k in ("R@1", "R@5", "R@10")))
