#!/bin/sh
# Финальный эксперимент, 4 камеры. $1 = gate | stream.
# gate: конфиги, маски динамики в корни, подготовка 5 фолдов × 2 варианта, проверка разбиений, побитная проверка старого прогона.
# stream: прогоны; порядок — сид 42 целиком, затем 43, 44; чекпойнты сида 42 сохраняются.
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/final4_runs"; UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/final4; mkdir -p "$OUT" "$KEEP"
wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 2000 ] && return 0; sleep 60; done; }
if [ "$1" = gate ]; then
  python3 experiments/generators/setup_final.py || { echo FINAL_FAIL; exit 1; }
  for root in rcg4_m2f_add_paddle rcg4_m2f_notext; do
    d="$HOME/tgseqloc_data/$root/robotcar/stages"
    if [ ! -d "$d/segmentation" ]; then mkdir -p "$d" && cp -r "$HOME/tgseqloc_data/rcgm_real_paddle/robotcar/stages/segmentation" "$d/"; fi
    echo "  маски в $root: $(ls $d/segmentation/base | wc -l) база, $(ls $d/segmentation/query | wc -l) запросы"
  done
  echo "=== подготовка $(date +%H:%M) ==="
  for arm in add notext; do
    ( for f in 0 1 2 3 4; do
        $UV run tgseqloc prepare -c "configs/final4_${arm}_f${f}_s42.yaml" > "$SC/final4_prep_${arm}_f$f.log" 2>&1 \
          || { echo "ОШИБКА подготовки $arm фолд $f"; tail -5 "$SC/final4_prep_${arm}_f$f.log"; exit 1; }
        echo "  $arm фолд $f готов $(date +%H:%M)"
      done ) &
  done
  wait
  python3 - <<'PY' || { echo FINAL_FAIL; exit 1; }
import json
from collections import Counter
from pathlib import Path
D = Path.home() / "tgseqloc_data"
cam = lambda p: Path(p).stem.split("-", 1)[0] if "-" in Path(p).stem else "stereo_centre"
bad = 0
for root in ("rcg4_m2f_add_paddle", "rcg4_m2f_notext"):
    for f in range(5):
        p = D / root / f"robotcar/mappings/split_fold{f}.json"
        if not p.exists():
            print("  НЕТ", p); bad += 1; continue
        s = json.load(open(p))
        q = s["query_paths"]
        parts = {k: Counter(cam(q[i]) for i in s[k + "_query_indices"]) for k in ("train", "validation", "test")}
        print("  %s фолд %d: база %s; запросы %s; тест %s; валидация %s; без позитивов %d" % (
            root, f, dict(Counter(cam(x) for x in s["database_paths"])), dict(Counter(cam(x) for x in q)),
            dict(parts["test"]), dict(parts["validation"]), s.get("queries_without_positives", -1)))
        bad += any(not parts[k] for k in parts)
a = json.load(open(D / "rcg4_m2f_add_paddle/robotcar/mappings/split_fold0.json"))
b = json.load(open(D / "rcg4_m2f_notext/robotcar/mappings/split_fold0.json"))
same = all(a[k] == b[k] for k in ("query_paths", "database_paths", "test_query_indices", "train_query_indices"))
print("  разбиения вариантов совпадают:", same); bad += not same
raise SystemExit(1 if bad else 0)
PY
  sed "s/^\(  experiment_name:\).*/\1 repro-final4-addp10-f0/" configs/f256_addp10_f0_s42.yaml > configs/repro_final4_addp10_f0.yaml
  $UV run tgseqloc train -c configs/repro_final4_addp10_f0.yaml > "$SC/repro_final4.log" 2>&1
  a=$(python3 "$SC/metrics_of.py" "$HOME/tgseqloc_data/f256_runs/addp10_f0_42.json"); b=$(python3 "$SC/metrics_of.py" outputs/repro-final4-addp10-f0/final_metrics.json)
  rm -f outputs/repro-final4-addp10-f0/*.pth
  if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "воспроизведение: совпало побитно ($b)"; touch "$OUT/ready"; else echo "воспроизведение: РАСХОЖДЕНИЕ [$a] → [$b]"; echo FINAL_FAIL; fi
  exit 0
fi
while [ ! -f "$OUT/ready" ]; do sleep 30; done
for s in ${FINAL_SEEDS:-42 43 44}; do for f in 0 1 2 3 4; do for arm in add notext; do
  key="${arm}_f${f}_s${s}"; name="final4-$arm-f$f-s$s"
  grep -q final_metrics "$OUT/$key.json" 2>/dev/null && continue
  mkdir "$OUT/.lock_$key" 2>/dev/null || continue
  wait_gpu; echo "=== $key $(date +%H:%M) ==="
  $UV run tgseqloc train -c "configs/final4_$key.yaml" > "$OUT/$key.json" 2>&1 || echo "  ОШИБКА $key"
  if [ "$s" = 42 ]; then mkdir -p "$KEEP/$name" && cp "outputs/$name/best_model.pth" "outputs/$name/final_metrics.json" "$KEEP/$name/" 2>/dev/null; fi
  rm -f "outputs/$name/best_model.pth" "outputs/$name/last_model.pth"
done; done; done
echo STREAM_DONE
