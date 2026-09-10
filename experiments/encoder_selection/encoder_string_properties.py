"""Все ли энкодеры заводятся и различают строки на обоих алфавитах."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "TGSeqLoc" / "src"))
import torch
from tgseqloc.preparation.text import build_char_ngram_encoder, build_multilingual_e5

LAT = ["CONFISERIE", "BOUCHERIE", "STARBUCKS", "PECLARD", "DEPOT"]
CYR = ["ГАРДЕРОБ", "ВЫХОД", "КВАНТ", "РЕКТОРАТ", "КАФЕДРА"]
NOISE = {"CONFISERIE": "CONFISER", "ГАРДЕРОБ": "ГAРДЕРОБ", "КВАНТ": "KBAHT"}

SPECS = [
    ("char_ngram 64", lambda: build_char_ngram_encoder(embedding_dim=64)),
    ("E5-small", lambda: build_multilingual_e5(
        model_name="intfloat/multilingual-e5-small", prefix="passage: ", pooling="mean")),
    ("E5-base", lambda: build_multilingual_e5(
        model_name="intfloat/multilingual-e5-base", prefix="passage: ", pooling="mean")),
    ("BGE-M3", lambda: build_multilingual_e5(
        model_name="BAAI/bge-m3", prefix="", pooling="cls")),
    ("LaBSE", lambda: build_multilingual_e5(
        model_name="sentence-transformers/LaBSE", prefix="", pooling="cls")),
    ("RuModernBERT", lambda: build_multilingual_e5(
        model_name="deepvk/RuModernBERT-base", prefix="", pooling="mean")),
]

def spread(enc, texts):
    """Медиана попарного косинуса разных строк: чем ниже, тем различимее."""
    v = enc.encode(texts)
    sims = [float(v[i] @ v[j]) for i in range(len(texts)) for j in range(i+1, len(texts))]
    return sum(sims) / len(sims)

print("%-14s %5s | %-16s | %-16s | устойчивость к ошибкам OCR"
      % ("энкодер", "dim", "латиница", "кириллица"))
for name, make in SPECS:
    try:
        enc = make()
        dim = int(enc.embedding_dim)
        lat, cyr = spread(enc, LAT), spread(enc, CYR)
        rob = []
        for good, bad in NOISE.items():
            v = enc.encode([good, bad])
            rob.append("%s→%s %.2f" % (good[:6], bad[:6], float(v[0] @ v[1])))
        print("%-14s %5d | медиана %.3f    | медиана %.3f    | %s"
              % (name, dim, lat, cyr, "  ".join(rob)))
    except Exception as exc:
        print("%-14s   —   | ОШИБКА: %s" % (name, str(exc)[:70]))
