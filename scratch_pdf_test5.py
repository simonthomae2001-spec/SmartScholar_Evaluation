import os
import time
from src.tools.pdf_tool import fetch_pdf_text, _cache_path

url = "https://arxiv.org/pdf/1706.03762"
pid = "attention-2017"

# 1. Lauf -> Download + Cache schreiben
t0 = time.time()
r1 = fetch_pdf_text(url, pid)
dt1 = time.time() - t0
print(f"1. Lauf: {len(r1.pages)} Seiten, {dt1:.2f}s (mit Download)")

# Cache-Datei sollte jetzt existieren
path = _cache_path(pid)
print("Cache-Datei existiert:", os.path.exists(path))
print("Pfad:", path)

# 2. Lauf -> Cache-Hit, KEIN Download
t1 = time.time()
r2 = fetch_pdf_text(url, pid)
dt2 = time.time() - t1
print(f"2. Lauf: {len(r2.pages)} Seiten, {dt2:.2f}s (aus Cache)")