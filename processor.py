import io
import json
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import pypdfium2 as pdfium

OPTION_LABELS = {
    1: "Não se aplica",
    2: "Nunca",
    3: "Por vezes",
    4: "Com frequência",
    5: "Sempre",
}

@dataclass
class CellDecision:
    page: int
    question: int
    option_index: Optional[int]
    option_label: str
    state: str
    confidence: float
    scores: List[float]

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def render_pdf_to_images(pdf_bytes: bytes, dpi: int):
    pdf = pdfium.PdfDocument(pdf_bytes)
    scale = dpi / 72.0
    pages = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
        arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        pages.append(arr)
    return pages

def preprocess_for_alignment(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def preprocess_binary(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
    )
    return bw

def align_to_model(image_bgr, model_bgr):
    img1 = preprocess_for_alignment(image_bgr)
    img2 = preprocess_for_alignment(model_bgr)

    orb = cv2.ORB_create(4000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return image_bgr.copy()

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 20:
        return image_bgr.copy()

    matches = sorted(matches, key=lambda x: x.distance)[:300]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        return image_bgr.copy()

    h, w = model_bgr.shape[:2]
    aligned = cv2.warpPerspective(image_bgr, H, (w, h), borderValue=(255, 255, 255))
    return aligned

def get_answer_zone(page_img, zone_cfg):
    h, w = page_img.shape[:2]
    x = int(zone_cfg["x"] * w)
    y = int(zone_cfg["y"] * h)
    ww = int(zone_cfg["w"] * w)
    hh = int(zone_cfg["h"] * h)
    return x, y, ww, hh

def compute_cell_boxes(zone_rect, cfg):
    x, y, w, h = zone_rect
    rows = cfg["questions_per_page"]
    cols = cfg["options_per_question"]

    top_margin = cfg["inner_margins"]["top"]
    bottom_margin = cfg["inner_margins"]["bottom"]
    left_margin = cfg["inner_margins"]["left"]
    right_margin = cfg["inner_margins"]["right"]

    yi1 = y + int(h * top_margin)
    yi2 = y + h - int(h * bottom_margin)
    xi1 = x + int(w * left_margin)
    xi2 = x + w - int(w * right_margin)

    inner_w = xi2 - xi1
    inner_h = yi2 - yi1

    cell_boxes = []
    for r in range(rows):
        row_boxes = []
        cy1 = yi1 + int(r * inner_h / rows)
        cy2 = yi1 + int((r + 1) * inner_h / rows)
        for c in range(cols):
            cx1 = xi1 + int(c * inner_w / cols)
            cx2 = xi1 + int((c + 1) * inner_w / cols)
            pad_x = int((cx2 - cx1) * cfg["cell_padding_x"])
            pad_y = int((cy2 - cy1) * cfg["cell_padding_y"])
            row_boxes.append((cx1 + pad_x, cy1 + pad_y, cx2 - pad_x, cy2 - pad_y))
        cell_boxes.append(row_boxes)
    return cell_boxes

def score_cell(aligned_bw, model_bw, box):
    x1, y1, x2, y2 = box
    cell = aligned_bw[y1:y2, x1:x2]
    model_cell = model_bw[y1:y2, x1:x2]
    added = cv2.bitwise_and(cell, cv2.bitwise_not(model_cell))
    score = float(np.count_nonzero(added)) / float(added.size if added.size else 1)
    return score

def decide_row(scores, cfg):
    best_idx = int(np.argmax(scores))
    best = scores[best_idx]
    second = sorted(scores, reverse=True)[1]

    mark_t = cfg["mark_threshold"]
    uncertain_t = cfg["uncertain_threshold"]
    margin_t = cfg["margin_threshold"]

    if best < uncertain_t:
        return None, "Sem resposta", 1.0 - min(best / max(uncertain_t, 1e-6), 1.0)

    high_count = sum(s >= mark_t for s in scores)
    if high_count >= 2:
        return None, "Múltipla", min(1.0, (best + second) / 2.0)

    if best < mark_t or (best - second) < margin_t:
        return best_idx + 1, "Duvidoso", min(1.0, best)

    return best_idx + 1, "OK", min(1.0, best)

def process_one_pdf(name, pdf_bytes, model_pages, cfg, debug=False):
    pages = render_pdf_to_images(pdf_bytes, cfg["dpi"])
    if len(pages) != cfg["pages_per_pdf"]:
        raise ValueError(f"{name}: esperado {cfg['pages_per_pdf']} páginas, recebido {len(pages)}.")

    decisions_all = []

    for page_idx in range(cfg["pages_per_pdf"]):
        model_bgr = model_pages[page_idx]
        filled_bgr = pages[page_idx]
        aligned_bgr = align_to_model(filled_bgr, model_bgr)

        model_bw = preprocess_binary(model_bgr)
        aligned_bw = preprocess_binary(aligned_bgr)

        zone_rect = get_answer_zone(aligned_bgr, cfg["page_zones"][page_idx])
        cell_boxes = compute_cell_boxes(zone_rect, cfg)

                if debug:
            import os
            os.makedirs("/tmp/questionario_debug", exist_ok=True)

            debug_img = aligned_bgr.copy()

            # desenhar retângulo azul da zona total de respostas
            zx, zy, zw, zh = zone_rect
            cv2.rectangle(debug_img, (zx, zy), (zx + zw, zy + zh), (255, 0, 0), 2)

            # desenhar cada célula em verde
            for row in cell_boxes:
                for box in row:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

            debug_name = f"{name.replace('.pdf', '')}_p{page_idx+1}.png"
            cv2.imwrite(f"/tmp/questionario_debug/{debug_name}", debug_img)

        for q in range(cfg["questions_per_page"]):
            scores = []
            for opt in range(cfg["options_per_question"]):
                score = score_cell(aligned_bw, model_bw, cell_boxes[q][opt])
                scores.append(score)

            opt_index, state, conf = decide_row(scores, cfg)

            decisions_all.append(
                CellDecision(
                    page=page_idx + 1,
                    question=q + 1,
                    option_index=opt_index,
                    option_label=OPTION_LABELS.get(opt_index, ""),
                    state=state,
                    confidence=conf,
                    scores=scores,
                )
            )

    return decisions_all

def build_excel(results):
    output = io.BytesIO()
    rows_wide = []
    rows_audit = []

    for filename, decisions in results.items():
        row = {"Ficheiro": filename}
        for d in decisions:
            key = f"P{d.page}_Q{d.question}"
            row[key] = d.option_label if d.option_label else d.state
            rows_audit.append({
                "Ficheiro": filename,
                "Página": d.page,
                "Pergunta": d.question,
                "Resposta": d.option_label if d.option_label else "",
                "Estado": d.state,
                "Confiança": round(d.confidence, 4),
                "Scores": ", ".join(f"{s:.4f}" for s in d.scores),
            })
        rows_wide.append(row)

    df_wide = pd.DataFrame(rows_wide)
    df_audit = pd.DataFrame(rows_audit)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_wide.to_excel(writer, index=False, sheet_name="Respostas")
        df_audit.to_excel(writer, index=False, sheet_name="Auditoria")

    output.seek(0)
    return output.getvalue()

    def process_uploaded_files(model_file, pdf_files, config_path: str, debug=False):
    cfg = load_config(config_path)
    model_pages = render_pdf_to_images(model_file.read(), cfg["dpi"])

    results = {}
    for f in pdf_files:
             results[f.filename] = process_one_pdf(f.filename, f.read(), model_pages, cfg, debug=debug)

        excel_bytes = build_excel(results)

    if debug:
        import os
        import zipfile

        zip_buffer = io.BytesIO()
        debug_folder = "/tmp/questionario_debug"

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            if os.path.exists(debug_folder):
                for fname in os.listdir(debug_folder):
                    full_path = os.path.join(debug_folder, fname)
                    zf.write(full_path, arcname=fname)

        zip_buffer.seek(0)
        return excel_bytes, zip_buffer.getvalue()

    return excel_bytes
