from src.tools.pdf_tool import fetch_pdf_text

url = "https://arxiv.org/pdf/1706.03762"   # "Attention Is All You Need"
res = fetch_pdf_text(url, "attention-2017")

print("Seiten mit Text:", len(res.pages))
print("has_content:", res.has_content)

if res.pages:
    seite_nr, text = res.pages[0]
    print(f"\n--- Seite {seite_nr}, erste 300 Zeichen ---")
    print(text[:300])

print("\nGesamtlaenge full_text():", len(res.full_text()), "Zeichen")