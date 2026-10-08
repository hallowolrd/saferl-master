"""Extract text from all PDFs in the paper directory for analysis."""
import os
import pdfplumber

PAPER_DIR = "paper"
OUTPUT_DIR = "temp/paper_texts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

papers = [f for f in os.listdir(PAPER_DIR) if f.endswith(".pdf")]

for paper in papers:
    pdf_path = os.path.join(PAPER_DIR, paper)
    txt_name = paper.replace(".pdf", ".txt")
    txt_path = os.path.join(OUTPUT_DIR, txt_name)

    if os.path.exists(txt_path):
        print(f"Skipping {paper} (already extracted)")
        continue

    print(f"Extracting: {paper}...")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"  Pages: {len(pdf.pages)}")
            full_text = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    full_text.append(f"=== PAGE {i+1} ===\n{text}")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(full_text))

            print(f"  Saved: {txt_path} ({len(full_text)} pages with text)")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone!")
