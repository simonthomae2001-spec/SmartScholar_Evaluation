from src.tools.pdf_tool import _download_pdf

# Echtes Open-Access-PDF (arXiv: "Attention Is All You Need") — stabil
url = "https://arxiv.org/pdf/1706.03762"
data = _download_pdf(url)

if data is None:
    print("❌ Download fehlgeschlagen (None).")
else:
    print(f"✅ {len(data)} bytes geladen")
    print("Erste Bytes:", data[:8])   # sollte mit b'%PDF' anfangen

# Negativ-Pfade: beide muessen sauber None liefern, kein Crash
print("Kaputte URL:", _download_pdf("not-a-real-url"))
print("Toter Link :", _download_pdf("https://arxiv.org/pdf/does-not-exist-999999"))