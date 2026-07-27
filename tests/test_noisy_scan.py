import conftest_path  # noqa: F401
import sys
import numpy as np, cv2
from core import markup_detect as md
from core import auto_ocr, doc_ir

rng = np.random.default_rng(42)

def paper(h, w, bg=225, noise=9):
    """Realistic scanned-paper background: gray, textured."""
    img = np.full((h, w), float(bg))
    img += rng.normal(0, noise, (h, w))
    # low-frequency shading gradient like a real scanner
    yy = np.linspace(-6, 6, h).reshape(-1, 1)
    img += yy
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def draw_box(img, x, y, s, ink=70, crossed=False, stroke=2):
    cv2.rectangle(img, (x, y), (x + s, y + s), (ink, ink, ink), 1)
    if crossed:
        d = int(s * 0.22)
        cv2.line(img, (x + d, y + d), (x + s - d, y + s - d), (ink, ink, ink), stroke)
        cv2.line(img, (x + s - d, y + d), (x + d, y + s - d), (ink, ink, ink), stroke)

def draw_circle(img, cx, cy, r, ink=70, crossed=False, stroke=2):
    cv2.circle(img, (cx, cy), r, (ink, ink, ink), 1)
    if crossed:
        d = int(r * 0.75)
        cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), (ink, ink, ink), stroke)
        cv2.line(img, (cx + d, cy - d), (cx - d, cy + d), (ink, ink, ink), stroke)

print("=== 1. single-mark crops on noisy scanned paper ===")
fails = 0
for bg, noise in ((235, 6), (225, 9), (210, 12)):
    for kind, crossed, expect in (("box", False, "unchecked"), ("box", True, "checked"),
                                  ("circle", False, "unchecked"), ("circle", True, "checked")):
        img = paper(46, 46, bg, noise)
        if kind == "box":
            draw_box(img, 6, 6, 34, crossed=crossed)
        else:
            draw_circle(img, 23, 23, 17, crossed=crossed)
        r = md.detect_markup_geometric(img)
        ok = r["state"] == expect
        if not ok: fails += 1
        print(f"bg={bg} noise={noise} {kind} crossed={crossed}: {r['state']} "
              f"(fill={r['fill_ratio']}, strike={r['strike']}) {'OK' if ok else '<-- FAIL'}")
assert fails == 0, f"{fails} noisy single-mark failures"
print("OK noisy single marks")

print("\n=== 2. proof: old Otsu approach fails on this data (diagnosis check) ===")
img = paper(46, 46, 225, 9); draw_box(img, 6, 6, 34, crossed=False)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
otsu_fill = np.count_nonzero(otsu[11:35, 11:35]) / otsu[11:35, 11:35].size
print(f"Otsu interior fill on EMPTY noisy box: {otsu_fill:.3f} (would read as checked: {otsu_fill >= 0.18})")

print("\n=== 3. survey row on noisy paper: one crossed of five ===")
row = paper(64, 520, 225, 9)
for i in range(5):
    draw_circle(row, 50 + i * 100, 32, 16, crossed=(i == 3))
marks = md.find_marks(row, n_expected=5)
states = [m["state"] for m in marks]
print("states:", states)
assert len(marks) == 5, f"found {len(marks)}"
assert states == ["unchecked", "unchecked", "unchecked", "checked", "unchecked"], states
print("OK discriminative row")

print("\n=== 4. end-to-end: disagreement is retained without OCR mutation ===")
page_img = paper(400, 600, 225, 9)
for i in range(5):
    draw_circle(page_img, 60 + i * 100, 120, 16, crossed=(i == 3))
region = doc_ir.Region(
    id="p1_r0", type=doc_ir.TEXT, bbox=[20, 80, 580, 165], reading_order=0,
    content={"text": "○ A ○ B ○ C ○ D ○ E".replace(" A ", "\t").replace(" B ", "\t").replace(" C ", "\t").replace(" D ", "\t").replace(" E ", "\t")},
)
region.content["text"] = "○\t○\t○\t○\t○"
page = doc_ir.Page(page_number=1, regions=[region], source="paddleocr-vl-1.6")
auto_ocr._apply_markup(page, page_img)
mk = region.markup
print("markup:", {k: mk.get(k) for k in ("status", "n_checked", "n_disagreements", "n_unchecked")})
assert mk["status"] == "geometry_disagreement", mk
assert mk["n_checked"] == 0 and mk["n_disagreements"] == 1, mk
assert region.content["text"].count("☒") == 0
print("unchanged:", region.content["text"])
print("OK disagreement retained on noisy paper")

print("\n=== 5. saturation guard: geometry claiming all-checked must not override ===")
glyphs = md.extract_mark_glyphs("○\t○\t○\t○\t○")
all_checked_geo = [{"state": "checked", "score": 0.9}] * 5
items, status = md.reconcile_marks(glyphs, all_checked_geo)
assert status == "geometry_saturated", status
assert all(i["state"] == "unchecked" for i in items)
print("OK saturation guard (status:", status + ")")
# A small group is also review-only by default; old override is benchmark-only.
items3, status3 = md.reconcile_marks(
    md.extract_mark_glyphs("○ ○"), [{"state": "checked", "score": 0.9}] * 2)
assert status3 == "geometry_disagreement" and all(i["state"] == "unchecked" for i in items3)
baseline3, _ = md.reconcile_marks(
    md.extract_mark_glyphs("○ ○"),
    [{"state": "checked", "score": 0.9}] * 2,
    allow_override=True,
)
assert all(i["state"] == "checked" for i in baseline3)
print("OK small-group override isolated to benchmark baseline")

print("\n=== 6. single '☐ label text' region on noisy paper -> NO override ===")
pg = paper(60, 500, 225, 9)
draw_box(pg, 10, 14, 30, crossed=False)
cv2.putText(pg, "Ich kenne die Angebote zu wenig", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
reg = doc_ir.Region(id="p1_r1", type=doc_ir.TEXT, bbox=[0, 0, 500, 60], reading_order=0,
                    content={"text": "☐ Ich kenne die Angebote zu wenig"})
pg_page = doc_ir.Page(page_number=1, regions=[reg], source="paddleocr-vl-1.6")
auto_ocr._apply_markup(pg_page, pg)
mk = reg.markup
print("markup:", {k: mk.get(k) for k in ("status", "n_checked", "n_overridden", "n_uncertain")})
assert mk["n_overridden"] == 0, mk
assert "☒" not in reg.content["text"]
print("OK empty single checkbox untouched")

print("\n=== 7. same region but genuinely crossed -> disagreement fires ===")
pg2 = paper(60, 500, 225, 9)
draw_box(pg2, 10, 14, 30, crossed=True)
cv2.putText(pg2, "Ich kenne die Angebote zu wenig", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
reg2 = doc_ir.Region(id="p1_r2", type=doc_ir.TEXT, bbox=[0, 0, 500, 60], reading_order=0,
                     content={"text": "☐ Ich kenne die Angebote zu wenig"})
pg2_page = doc_ir.Page(page_number=1, regions=[reg2], source="paddleocr-vl-1.6")
auto_ocr._apply_markup(pg2_page, pg2)
mk2 = reg2.markup
print("markup:", {k: mk2.get(k) for k in ("status", "n_checked", "n_disagreements")})
assert mk2["n_disagreements"] == 1 and reg2.content["text"].startswith("☐"), (mk2, reg2.content)
print("OK genuine cross flagged without mutation:", reg2.content["text"])

print("\nALL NOISY-SCAN TESTS PASSED")
