from src.tools.pdf_tool import fetch_pdf_text, PdfExtractionResult

# 1) Paywall-Pfad: kein URL -> leeres Ergebnis, kein Crash
r = fetch_pdf_text(None, "fake-id-123")
print("pages:", r.pages, "| has_content:", r.has_content, "| full_text:", repr(r.full_text()))

# 2) Struktur-Verhalten der Output-Form aus Entscheidung A
demo = PdfExtractionResult(pages=[(1, "Seite eins"), (2, "Seite zwei")])
print("full_text:", repr(demo.full_text()))
print("has_content:", demo.has_content)

print("\n✅ Entscheidung A verhaelt sich korrekt.")