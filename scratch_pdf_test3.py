from src.tools.pdf_tool import _looks_like_pdf, fetch_pdf_text

# Unit: die Magic-Byte-Prüfung selbst
print("PDF bytes :", _looks_like_pdf(b"%PDF-1.5\n..."))         # True
print("HTML bytes:", _looks_like_pdf(b"<!DOCTYPE html><html>")) # False
print("Empty     :", _looks_like_pdf(b""))                      # False

# Integration: eine HTML-Seite liefert HTTP 200, ist aber KEIN PDF.
# -> C faengt das ab, bevor der (noch nicht gebaute) Parser E erreicht wird.
html_url = "https://arxiv.org/abs/1706.03762"   # die HTML-Abstract-Seite
res = fetch_pdf_text(html_url, "html-test")
print("HTML-Seite -> pages:", res.pages, "| has_content:", res.has_content)