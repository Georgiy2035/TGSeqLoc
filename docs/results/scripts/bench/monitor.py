"""Каждые 0.25 с: загрузка GPU, занятая видеопамять, загрузка CPU (все ядра) — пока есть файл-флаг."""
import subprocess, sys, time
from pathlib import Path
import psutil
flag, out = Path(sys.argv[1]), open(sys.argv[2], "w")
out.write("t,gpu_util,gpu_mem_mb,cpu_percent\n"); psutil.cpu_percent(None)
while flag.exists():
    time.sleep(0.25)
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split(",")
    out.write("%.3f,%s,%s,%.1f\n" % (time.time(), q[0].strip(), q[1].strip(), psutil.cpu_percent(None))); out.flush()
