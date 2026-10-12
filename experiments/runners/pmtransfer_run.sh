#!/bin/sh
# Первомайская без обучения, веса финального RobotCar. $1 = gate | eval.
# gate: конфиги, маски динамики в корни, подготовка трёх корней (текст, без текста, перемешанный текст).
# eval: 40 оценок (final4 и final4s × add, notext, textoff, shuf × фолды 0–4), по 4 одновременно.
OUT="$HOME/tgseqloc_data/pmtransfer_runs"; BIN="$HOME/TGSeqLoc/.venv/bin"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints; mkdir -p "$OUT/logs"
if [ "$1" = gate ]; then
  "$BIN/python" experiments/generators/setup_pervomay.py configs || { echo PM_FAIL; exit 1; }
  "$BIN/python" experiments/generators/setup_pervomay.py masks || { echo PM_FAIL; exit 1; }
  echo "=== подготовка $(date +%H:%M) ==="
  for arm in add notext shuf; do
    ( "$BIN/tgseqloc" prepare -c "configs/pm_prepare_${arm}.yaml" > "$OUT/logs/prepare_${arm}.log" 2>&1 \
        && echo "  $arm готов $(date +%H:%M)" || { echo "ОШИБКА подготовки $arm"; tail -5 "$OUT/logs/prepare_${arm}.log"; } ) &
  done
  wait
  echo PM_GATE_DONE
elif [ "$1" = eval ]; then
  for w in final4 final4s; do
    for e in add notext textoff shuf; do
      case $e in notext) arm=notext;; *) arm=add;; esac
      for f in 0 1 2 3 4; do
        echo "$w $e $f $KEEP/$w/$w-$arm-f$f-s42/best_model.pth"
      done
    done
  done | xargs -P 4 -L 1 sh -c '
    name="pm-$0-$1-f$2"
    if [ -f "'"$OUT"'/$name/final_metrics.json" ]; then exit 0; fi
    "'"$BIN"'/tgseqloc" transfer -c "configs/pm_$0_$1_f$2.yaml" --checkpoint "$3" > "'"$OUT"'/logs/$name.log" 2>&1 \
      && echo "  $name $(date +%H:%M:%S)" || echo "  ОШИБКА $name"'
  echo PM_EVAL_DONE
fi
