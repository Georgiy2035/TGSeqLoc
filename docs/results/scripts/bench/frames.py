"""500 тестовых запросов фолда 0 финального эксперимента: по 125 на камеру, равномерно по времени."""
import json
from pathlib import Path
D = Path.home() / "tgseqloc_data"; T = "2015-09-02-10-37-32"
s = json.load(open(D / "rcg4_m2f_add_paddle/robotcar/mappings/split_fold0.json"))
cam_of = lambda st: st.split("-", 1)[0] if "-" in st else "stereo_centre"
by = {}
for i in s["test_query_indices"]:
    st = Path(s["query_paths"][i]).stem; by.setdefault(cam_of(st), []).append(st)
out = []
for cam, stems in sorted(by.items()):
    stems.sort(key=lambda x: int(x.split("-")[-1])); step = len(stems) / 125
    for k in range(125):
        st = stems[int(k * step)]; ts = st.split("-")[-1]
        ocr = (f"{Path.home()}/ocr_benchmark/results/robotcar_{T}/{ts}/paddleocr_v5.json" if cam == "stereo_centre"
               else f"/mnt/external_usb_hdd/6YL/sitnikov_sv/robotcar_mono_ocr/robotcar_{T}_{cam}/{ts}/paddleocr_v5.json")
        out.append({"camera": cam, "stem": st, "image": f"/mnt/external_usb_hdd/6YL/Datasets/robotcar/{T}_rgb/{cam}/{ts}.png",
                    "ocr": ocr, "graph": f"/mnt/external_usb_hdd/6YL/Datasets/scene_graphs_robotcar_m2f/{T}_rgb/{cam}/{ts}.json",
                    "mask": str(D / f"rcg4_m2f_add_paddle/robotcar/stages/segmentation/query/{st}.json")})
for f in out:
    for k in ("image", "ocr", "graph", "mask"):
        assert Path(f[k]).exists(), f[k]
json.dump(out, open(Path(__file__).with_name("frames.json"), "w"), indent=1)
print("кадров:", len(out), {c: sum(1 for f in out if f["camera"] == c) for c in by})
