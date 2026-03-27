import pdfplumber
import json
import re


def is_class_header(row):
    """
    A class header row has exactly one non-empty cell (or all identical cells
    because pdfplumber sometimes repeats merged cell content).
    It contains a class name with a digit count, e.g. "Alkaloids (2)".
    """
    cells = [c for c in row if c and c.strip()]
    if not cells:
        return False
    unique = set(c.strip() for c in cells)
    # All cells the same value (merged cell repeated) or only one cell present
    if len(unique) != 1:
        return False
    value = cells[0].strip()
    return bool(re.search(r'\(\d+\)', value))

def clean_class_name(raw):
    """Remove ' continued' and the count '(N)' to get a lookup key."""
    name = re.sub(r'\s+continued\s*$', '', raw, flags=re.IGNORECASE).strip()
    name = re.sub(r'\s*\(\d+\)', '', name).strip()
    return name

# ── Step 1: Build class → method mapping from pages 2–3 ──────────────────────
#
# Structure on these pages:
#   Row 1 (merged header): "Analyte class (number of metabolites)" | "Analytical method"
#   Row 2 (merged group):  "Small molecules (327)"                 |
#   Row 3 (subclass):      "Alkaloids (2)"                         | "LC-MS/MS"
#   ...
#   Row N (merged group):  "Lipids (906)"                          |
#   Row N+1 (subclass):    "Acylcarnitines (40)"                   | "FIA-MS/MS"
#
# Structure:
#   Header row:  single cell spanning all columns, e.g. "Alkaloids (2)"
#                OR a continuation header, e.g. "Amino acid-related (77) continued"
#   Data rows:   4 columns: abbrev1 | full_name1 | abbrev2 | full_name2
#                Some data rows have only 2 populated columns (last row of a class)
def get_class_details(pdf):
  for page_idx in METHOD_PAGES:
      page = pdf.pages[page_idx]
      tables = page.extract_tables()
      for table in tables:
          for row in table:
              # Flatten row: drop None/empty cells
              cells = [c.strip() for c in row if c and c.strip()]
              if not cells:
                  continue

              # Detect a method cell anywhere in the row
              if any(c in ("LC-MS/MS", "FIA-MS/MS") for c in cells):
                  current_method = next(
                      c for c in cells if c in ("LC-MS/MS", "FIA-MS/MS")
                  )

              # Detect a class name: contains digits in parentheses, e.g. "Alkaloids (2)"
              # but is NOT the top-level group header ("Small molecules (327)" etc.)
              # We exclude the two group headers explicitly.
              GROUP_HEADERS = {
                  "Small molecules (327)", "Lipids (906)",
                  "Analyte class (number of metabolites)"
              }
              for cell in cells:
                  if (re.search(r'\(\d+\)', cell)
                          and cell not in GROUP_HEADERS
                          and current_method):
                      # Strip the count to get a clean class name key
                      clean = re.sub(r'\s*\(\d+\)', '', cell).strip()
                      method_map[clean] = current_method
                      print(f"Mapped {len(method_map)} classes to methods")

def get_details(pdf):
  for page_idx in DETAIL_PAGES:
      page = pdf.pages[page_idx]
      tables = page.extract_tables()
      for table in tables:
          for row in table:
              if is_class_header(row):
                  raw_name = [c for c in row if c and c.strip()][0].strip()
                  current_class = clean_class_name(raw_name)
                  continue
              if current_class is None:
                  continue
              # Pad row to at least 4 elements
              padded = list(row) + [None] * 4
              a1, n1, a2, n2 = (
                  (padded[0] or "").strip(),
                  (padded[1] or "").strip(),
                  (padded[2] or "").strip(),
                  (padded[3] or "").strip(),
              )
              method = method_map.get(current_class, "Unknown")
              if a1 and n1:
                  metabolites.append({
                      "abbreviation": a1,
                      "full_name":    n1,
                      "class":        current_class,
                      "method":       method,
                  })
              if a2 and n2:
                  metabolites.append({
                      "abbreviation": a2,
                      "full_name":    n2,
                      "class":        current_class,
                      "method":       method,
                  })



METHOD_PAGES = [1, 2]   # 0-indexed: PDF pages 2–3
method_map = {}         # e.g. {"Alkaloids": "LC-MS/MS", "Acylcarnitines": "FIA-MS/MS"}
current_method = None

DETAIL_PAGES = list(range(3, 19))   # 0-indexed: PDF pages 4–19

metabolites = []
current_class = None


with pdfplumber.open("data/metabolites.pdf") as pdf:
    get_class_details(pdf)
    get_details(pdf)


# ── Step 3: Sanity check ──────────────────────────────────────────────────────
from collections import Counter

class_counts = Counter(m["class"] for m in metabolites)
print("\nClass counts:")
for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    expected_method = method_map.get(cls, "?")
    print(f"  {cls:45s} {count:4d}   [{expected_method}]")

missing_method = [m for m in metabolites if m["method"] == "Unknown"]
if missing_method:
    print(f"\nWARNING: {len(missing_method)} metabolites with no method mapping:")
    for m in missing_method[:5]:
        print(f"  {m['class']} / {m['abbreviation']}")


# ── Step 4: Save ──────────────────────────────────────────────────────────────
with open("data/metabolites.json", "w") as f:
    json.dump(metabolites, f, indent=2, ensure_ascii=False)

