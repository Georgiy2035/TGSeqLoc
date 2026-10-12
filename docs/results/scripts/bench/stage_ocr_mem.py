"""OCR без чтения кадра: все кадры заранее декодируются в память (PIL RGB, как load_oriented_image),
а загрузка кадра внутри модели подменяется выдачей готового изображения. Пакет 1, 5 кадров прогрева."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.home() / "ocr_benchmark"))
from benchmark.image import load_oriented_image
from benchmark.registry import create_model
from benchmark.runner import _configure_model_storage
import benchmark.models.paddleocr as m_paddle
import benchmark.models.qwen as m_qwen
code, n, log = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
frames = json.load(open(Path(__file__).with_name("frames.json")))
if n < len(frames):
    frames = [frames[int(i * len(frames) / n)] for i in range(n)]
cache = {f["image"]: load_oriented_image(Path(f["image"])) for f in frames}      # чтение и декодирование — вне замера
from_memory = lambda p: cache[str(p)]
m_paddle.load_oriented_image = from_memory; m_qwen.load_oriented_image = from_memory
storage = Path.home() / "ocr_benchmark/.model_storage"; _configure_model_storage(storage)
model = create_model(model_code=code, use_gpu=True, model_dir=storage); model.load()
for f in frames[:5]:
    model.predict(Path(f["image"]))
try:
    import torch; torch.cuda.reset_peak_memory_stats(); has_torch = torch.cuda.is_available()
except Exception:
    has_torch = False
start = time.time(); times = []
for f in frames[5:]:
    t0 = time.perf_counter(); model.predict(Path(f["image"])); times.append(1000 * (time.perf_counter() - t0))
end = time.time()
peak = torch.cuda.max_memory_allocated() / 2**20 if has_torch else None
with open(log, "a") as fh:
    fh.write(json.dumps({"stage": f"OCR {code} (без чтения кадра)", "device": "GPU", "start": start, "end": end, "times_ms": times, "torch_peak_mb": peak}) + "\n")
print(code, "готово", len(times))
