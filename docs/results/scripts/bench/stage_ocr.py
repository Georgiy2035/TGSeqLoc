"""OCR по кадрам, пакет 1: время predict на кадр (первые 5 кадров — прогрев, не учитываются)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.home() / "ocr_benchmark"))
from benchmark.registry import create_model
from benchmark.runner import _configure_model_storage
code, n, log = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
frames = json.load(open(Path(__file__).with_name("frames.json")))
if n < len(frames):
    frames = [frames[int(i * len(frames) / n)] for i in range(n)]
storage = Path.home() / "ocr_benchmark/.model_storage"; _configure_model_storage(storage)
model = create_model(model_code=code, use_gpu=True, model_dir=storage); model.load()
for f in frames[:5]:
    model.predict(Path(f["image"]))
peak = None
try:
    import torch; torch.cuda.reset_peak_memory_stats()
except Exception:
    torch = None
start = time.time(); times = []
for f in frames[5:]:
    t0 = time.perf_counter(); model.predict(Path(f["image"])); times.append(1000 * (time.perf_counter() - t0))
end = time.time()
if torch is not None and torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated() / 2**20
with open(log, "a") as fh:
    fh.write(json.dumps({"stage": f"OCR {code}", "device": "GPU", "start": start, "end": end, "times_ms": times, "torch_peak_mb": peak}) + "\n")
print(code, "готово", len(times))
