from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import fitz  # PyMuPDF
import io
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imutils
import math
import pytesseract
import sys
import os

# Intelligently find Tesseract-OCR based on the OS
if sys.platform == "win32":
    # --- LOCAL DEVELOPMENT (WINDOWS) ---
    tess_paths = [
        os.path.join(os.getcwd(), "Tesseract-OCR", "tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
    ]
    for p in tess_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break
else:
    # --- CLOUD DEPLOYMENT (LINUX/MAC) ---
    # In cloud environments, Tesseract is installed via the system package manager
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

    # Note: You must configure your cloud provider to install tesseract-ocr
    # e.g., via a Dockerfile or a buildpack.

import re
import traceback
from typing import List

from rapidfuzz import fuzz
from detect import run_detection_pil, get_center

try:
    from Extract import extract_all_features
except ImportError as e:
    raise RuntimeError(f"Error loading external module: {e}")

app = FastAPI(title="Label Comparator API", description="Production-ready FastAPI application for Label Comparator")

@app.get("/api/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "ocr": "unknown",
        "4sym_model": "unknown",
        "16sym_model": "unknown"
    }

    # Test OCR
    try:
        pytesseract.get_tesseract_version()
        health_status["ocr"] = "working"
    except Exception as e:
        health_status["ocr"] = f"failed: {str(e)}"
        health_status["status"] = "degraded"

    # Test YOLO models
    model_4sym_path = os.path.join("4sym_models", "best.pt")
    model_16sym_path = os.path.join("16sym_models", "best.pt")

    if os.path.exists(model_4sym_path):
        health_status["4sym_model"] = "working"
    else:
        health_status["4sym_model"] = "missing"
        health_status["status"] = "degraded"

    if os.path.exists(model_16sym_path):
        health_status["16sym_model"] = "working"
    else:
        health_status["16sym_model"] = "missing"
        health_status["status"] = "degraded"

    return JSONResponse(status_code=200 if health_status["status"] == "healthy" else 503, content=health_status)

# Setup CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def pdf_to_image(file_bytes: bytes, dpi=200):
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    first_page = pdf_document[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = first_page.get_pixmap(matrix=mat, alpha=False)
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pdf_document.close()
    return image

async def process_upload(uploaded_file: UploadFile, max_width=1000):
    if not uploaded_file:
        return None
    file_bytes = await uploaded_file.read()
    if uploaded_file.filename.lower().endswith(".pdf"):
        img = pdf_to_image(file_bytes)
    else:
        img = Image.open(io.BytesIO(file_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
    if img.width > max_width:
        ratio = max_width / float(img.width)
        new_height = int((float(img.height) * float(ratio)))
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
    return img

def preprocess_image(image, resize_to=None, enhance_contrast=False):
    if image is None: return None
    img = image.copy()
    if resize_to:
        img = img.resize(resize_to, Image.Resampling.LANCZOS)
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
    return img

def align_images(imageA, imageB, max_features=500, good_match_percent=0.15):
    try:
        grayA = cv2.cvtColor(np.array(imageA), cv2.COLOR_RGB2GRAY)
        grayB = cv2.cvtColor(np.array(imageB), cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(max_features)
        keypointsA, descriptorsA = orb.detectAndCompute(grayA, None)
        keypointsB, descriptorsB = orb.detectAndCompute(grayB, None)
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptorsA, descriptorsB)
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * good_match_percent)
        matches = matches[:numGoodMatches]
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypointsA[match.queryIdx].pt
            points2[i, :] = keypointsB[match.trainIdx].pt
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        height, width = grayB.shape[:2]
        aligned = cv2.warpPerspective(np.array(imageA), h, (width, height))
        return Image.fromarray(aligned), True
    except Exception as e:
        return imageA, False

def find_differences(imageA, imageB, threshold=0.85, min_area=150):
    try:
        if imageA.size != imageB.size:
            imageB = imageB.resize(imageA.size, Image.Resampling.LANCZOS)
        grayA = cv2.cvtColor(np.array(imageA), cv2.COLOR_RGB2GRAY)
        grayB = cv2.cvtColor(np.array(imageB), cv2.COLOR_RGB2GRAY)
        score, diff = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        filtered_cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
        bounding_boxes = []
        for c in filtered_cnts:
            x, y, w, h = cv2.boundingRect(c)
            bounding_boxes.append((x, y, w, h))
        return {
            'ssim_score': score,
            'bounding_boxes': bounding_boxes,
            'total_differences': len(bounding_boxes)
        }
    except Exception as e:
        print(f"Error finding differences: {e}")
        return None

def boxes_overlap(boxA, boxB, threshold=0.3):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    smaller_area = min(boxAArea, boxBArea)
    if smaller_area == 0: return False
    return (interArea / float(smaller_area)) > threshold

def filter_text_boxes(text_boxes, symbol_boxes):
    filtered = []
    for (x, y, w, h) in text_boxes:
        text_box = [x, y, x + w, y + h]
        overlap = False
        for sym in symbol_boxes:
            sym_box = sym["bbox"]
            if boxes_overlap(text_box, sym_box):
                overlap = True
                break
        if not overlap:
            filtered.append((x, y, w, h))
    return filtered

def draw_differences(image, bounding_boxes, color=(255, 0, 0), thickness=2, label=""):
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    for (x, y, w, h) in bounding_boxes:
        draw.rectangle([x, y, x + w, y + h], outline=color, width=thickness)
        if label:
            draw.text((x, max(0, y-15)), label, fill=color)
    return img_with_boxes

def draw_symbol_boxes(image, detections, color_map=None, thickness=2):
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    if color_map is None:
        color_map = {
            "Added": (0,255,0), 
            "Removed": (255,0,0), 
            "Misplaced": (255,165,0),  
            "Symbol": (0,0,255) 
        }
        
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = d.get("label", "Symbol")
        color = color_map.get(label, (0,0,255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        draw.text((x1, max(0, y1-15)), label, fill=color)
    return img_with_boxes

def get_feature_diffs(base_df, comp_df, comp_type, fuzzy_threshold=85):
    if base_df.empty or comp_df.empty:
        return [], []
    
    base_vals_list = base_df[base_df['Type'] == comp_type]['Value'].tolist()
    comp_vals_list = comp_df[comp_df['Type'] == comp_type]['Value'].tolist()
    
    added = []
    deleted = []

    if comp_type in ['Barcode', 'Image']:
        base_set = set(base_vals_list)
        comp_set = set(comp_vals_list)
        added = list(comp_set - base_set)
        deleted = list(base_set - comp_set)
        return added, deleted

    # Smarter token-based matching to ignore OCR noise
    for b_val in base_vals_list:
        match_found = False
        norm_b = b_val.lower().strip() 
        for c_val in comp_vals_list:
            if fuzz.token_set_ratio(norm_b, c_val.lower().strip()) >= fuzzy_threshold:
                match_found = True
                break
        if not match_found:
            deleted.append(b_val)

    for c_val in comp_vals_list:
        match_found = False
        norm_c = c_val.lower().strip() 
        for b_val in base_vals_list:
            if fuzz.token_set_ratio(norm_c, b_val.lower().strip()) >= fuzzy_threshold:
                match_found = True
                break
        if not match_found:
            added.append(c_val)

    return added, deleted

def ocr_crop(image, box):
    """Extracts text ONLY from the exact physical area that changed"""
    x, y, w, h = box
    pad = 5
    
    img_width = image.shape[1] if isinstance(image, np.ndarray) else image.width
    img_height = image.shape[0] if isinstance(image, np.ndarray) else image.height
    
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(img_width, x+w+pad), min(img_height, y+h+pad)
    
    crop = np.array(image)[y1:y2, x1:x2]
    if crop.size == 0: return ""
    
    if len(crop.shape) == 3: gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else: gray = crop
        
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(gray, lang='eng+fra+deu', config='--psm 6').strip()
    
    text = re.sub(r'[|><_~=«»"*;]', '', text).strip()
    text = re.sub(r'\n+', ' ', text) 
    return text

@app.post("/api/compare")
async def compare_labels(
    base_file: UploadFile = File(...),
    child_files: List[UploadFile] = File(...)
):
    if not base_file or not child_files:
        raise HTTPException(status_code=400, detail="Please ensure both the Base Label and at least one Child Label are uploaded.")
        
    try:
        raw_base_img = await process_upload(base_file)
        base_processed = preprocess_image(raw_base_img, enhance_contrast=False)
        
        # Run YOLO once and pass it down
        base_symbols_raw = run_detection_pil(base_processed)
        base_features_df = extract_all_features(raw_base_img, base_symbols_raw, logo_folder="logos")
        
        base_symbols = []
        for d in base_symbols_raw:
            d = d.copy()
            d["label"] = "Symbol"
            base_symbols.append(d)
            
        final_results = []
        
        for child_file in child_files:
            raw_child_img = await process_upload(child_file)
            comp_processed = preprocess_image(raw_child_img, enhance_contrast=False)
            
            comp_aligned, aligned_success = align_images(base_processed, comp_processed)
            if not aligned_success:
                comp_aligned = comp_processed
            
            # Run YOLO on child once and pass it down
            comp_symbols_raw = run_detection_pil(comp_aligned)
            comp_features_df = extract_all_features(comp_aligned, comp_symbols_raw, logo_folder="logos")
                
            diff_results = find_differences(base_processed, comp_aligned, threshold=0.85, min_area=150)
            
            if not diff_results:
                final_results.append({
                    "filename": child_file.filename,
                    "error": "Error comparing document"
                })
                continue
                
            comp_symbols_final = []
            
            def is_box_image(box, features_df):
                if features_df.empty: return False
                img_rows = features_df[features_df['Type'] == 'Image']
                for _, row in img_rows.iterrows():
                    sym_box = row.get("Box")
                    if hasattr(sym_box, '__len__') and len(sym_box) == 4 and boxes_overlap(box, sym_box, threshold=0.01):
                        return True
                return False

            def region_has_symbol(image, bbox, threshold=15):
                x1, y1, x2, y2 = map(int, bbox)
                crop = np.array(image)[y1:y2, x1:x2]
                if crop.size == 0: return False
                if len(crop.shape) == 3:
                    crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                else:
                    crop_gray = crop
                non_bg = np.sum(crop_gray < 240)
                return non_bg > threshold

            for base_sym in base_symbols_raw:
                if is_box_image(base_sym["bbox"], base_features_df): continue
                matches = [c for c in comp_symbols_raw if c["class"] == base_sym["class"]]
                if matches:
                    for match in matches:
                        if is_box_image(match["bbox"], comp_features_df): continue
                        c1 = get_center(base_sym["bbox"])
                        c2 = get_center(match["bbox"])
                        dist = math.dist(c1, c2)
                        if dist > 40: 
                            if region_has_symbol(comp_aligned, match["bbox"]):
                                misplaced_box = match.copy()
                                misplaced_box["label"] = "Misplaced"
                                comp_symbols_final.append(misplaced_box)
                else:
                    missing_box = base_sym.copy()
                    missing_box["label"] = "Removed"
                    comp_symbols_final.append(missing_box)
                    
            for d in comp_symbols_raw:
                if is_box_image(d["bbox"], comp_features_df): continue
                if d["class"] not in [b["class"] for b in base_symbols_raw]:
                    added_box = d.copy()
                    added_box["label"] = "Added"
                    comp_symbols_final.append(added_box)
                    
            comp_symbols = comp_symbols_final
            
            ssim_boxes = diff_results['bounding_boxes']
            text_diff_boxes = []
            for box in ssim_boxes:
                overlap = False
                box_coords = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
                for sym in base_symbols_raw:
                    if boxes_overlap(box_coords, sym["bbox"]): overlap = True; break
                for sym in comp_symbols_raw:
                    if boxes_overlap(box_coords, sym["bbox"]): overlap = True; break
                if not overlap:
                    text_diff_boxes.append(box)

            actual_deleted_boxes = []
            actual_added_boxes = []
            changed_boxes = []

            base_gray = cv2.cvtColor(np.array(base_processed), cv2.COLOR_RGB2GRAY)
            child_gray = cv2.cvtColor(np.array(comp_aligned), cv2.COLOR_RGB2GRAY)

            for (x, y, w, h) in text_diff_boxes:
                box_coords = [x, y, x+w, y+h]
                if is_box_image(box_coords, base_features_df) or is_box_image(box_coords, comp_features_df):
                    continue # Skip text diffing for image regions
                    
                crop_b = base_gray[y:y+h, x:x+w]
                crop_c = child_gray[y:y+h, x:x+w]
                
                if crop_b.size == 0 or crop_c.size == 0: continue
                
                dark_pixels_b = np.sum(crop_b < 220)
                dark_pixels_c = np.sum(crop_c < 220)
                
                min_pixels = 15 
                has_content_b = dark_pixels_b > min_pixels
                has_content_c = dark_pixels_c > min_pixels
                
                if has_content_b and not has_content_c:
                    actual_deleted_boxes.append((x, y, w, h))
                elif not has_content_b and has_content_c:
                    actual_added_boxes.append((x, y, w, h))
                elif has_content_b and has_content_c:
                    changed_boxes.append((x, y, w, h))

            added_text = []
            for box in actual_added_boxes:
                if is_box_image(box, comp_features_df): continue
                txt = ocr_crop(comp_aligned, box)
                if txt and len(txt) > 2: added_text.append(txt)

            deleted_text = []
            for box in actual_deleted_boxes:
                if is_box_image(box, base_features_df): continue
                txt = ocr_crop(base_processed, box)
                if txt and len(txt) > 2: deleted_text.append(txt)

            modified_text = []
            for box in changed_boxes:
                if is_box_image(box, base_features_df) or is_box_image(box, comp_features_df): continue
                txt_b = ocr_crop(base_processed, box)
                txt_c = ocr_crop(comp_aligned, box)
                if txt_b or txt_c:
                    modified_text.append(f"From: '{txt_b}' ➔ To: '{txt_c}'")

            added_bc, deleted_bc = get_feature_diffs(base_features_df, comp_features_df, 'Barcode')
            added_img, deleted_img = get_feature_diffs(base_features_df, comp_features_df, 'Image')

            modified_image_details = []
            modified_image_boxes = []
            
            while added_img and deleted_img:
                a_img_val = added_img.pop(0)
                d_img_val = deleted_img.pop(0)
                modified_image_details.append(f"From: '{d_img_val}' ➔ To: '{a_img_val}'")
                print(f"DEBUG: Processing Image Modification from {d_img_val} to {a_img_val}")
                b_box = base_features_df[(base_features_df['Type'] == 'Image') & (base_features_df['Value'] == d_img_val)].iloc[0]['Box']
                c_box = comp_features_df[(comp_features_df['Type'] == 'Image') & (comp_features_df['Value'] == a_img_val)].iloc[0]['Box']
                print(f"DEBUG: Found boxes: B={b_box}, C={c_box}")
                modified_image_boxes.append((b_box, c_box))

            def fetch_boxes(df, f_type, val_list):
                boxes = []
                for v in val_list:
                    try:
                        b = df[(df['Type'] == f_type) & (df['Value'] == v)].iloc[0]['Box']
                        boxes.append(b)
                    except: pass
                return boxes
                
            added_img_boxes = fetch_boxes(comp_features_df, 'Image', added_img)
            deleted_img_boxes = fetch_boxes(base_features_df, 'Image', deleted_img)
            added_bc_boxes = fetch_boxes(comp_features_df, 'Barcode', added_bc)
            deleted_bc_boxes = fetch_boxes(base_features_df, 'Barcode', deleted_bc)

            added_syms = [s["class"] for s in comp_symbols if s["label"] == "Added"]
            removed_syms = [s["class"] for s in comp_symbols if s["label"] == "Removed"]
            misplaced_syms = [s["class"] for s in comp_symbols if s["label"] == "Misplaced"]
            
            discrepancy_report = {
                "Added": [],
                "Deleted": [],
                "Modified": [],
                "Misplaced": []
            }
            
            for item in added_text: discrepancy_report["Added"].append({"Category": "Text", "Value": item})
            for item in deleted_text: discrepancy_report["Deleted"].append({"Category": "Text", "Value": item})
            for item in modified_text: discrepancy_report["Modified"].append({"Category": "Text", "Value": item})
            
            for item in added_syms: discrepancy_report["Added"].append({"Category": "Symbol", "Value": item})
            for item in misplaced_syms: discrepancy_report["Misplaced"].append({"Category": "Symbol", "Value": item})
            for item in removed_syms: discrepancy_report["Deleted"].append({"Category": "Symbol", "Value": item})
            
            for item in added_bc: discrepancy_report["Added"].append({"Category": "Barcode", "Value": item})
            for item in deleted_bc: discrepancy_report["Deleted"].append({"Category": "Barcode", "Value": item})
            
            for item in added_img: discrepancy_report["Added"].append({"Category": "Image", "Value": item})
            for item in deleted_img: discrepancy_report["Deleted"].append({"Category": "Image", "Value": item})
            for item in modified_image_details: discrepancy_report["Modified"].append({"Category": "Image", "Value": item})

            base_features_records = base_features_df.to_dict(orient="records") if not base_features_df.empty else []
            comp_features_records = comp_features_df.to_dict(orient="records") if not comp_features_df.empty else []

            # ---- DRAW BOUNDING BOXES ----
            base_draw_actions = []
            child_draw_actions = []

            color_added = (21, 128, 61)     # Green
            color_deleted = (222, 38, 38)   # Red
            color_modified = (30, 61, 137)  # Dark Blue
            color_misplaced = (245, 163, 10) # Orange

            for box in actual_deleted_boxes:
                base_draw_actions.append((box, color_deleted, "Deleted", True))
            for box in actual_added_boxes:
                child_draw_actions.append((box, color_added, "Added", True))
            for box in changed_boxes:
                base_draw_actions.append((box, color_modified, "Modified", True))
                child_draw_actions.append((box, color_modified, "Modified", True))

            for base_sym in base_symbols_raw:
                matches = [c for c in comp_symbols_raw if c["class"] == base_sym["class"]]
                if matches:
                    for match in matches:
                        c1 = get_center(base_sym["bbox"])
                        c2 = get_center(match["bbox"])
                        dist = math.dist(c1, c2)
                        if dist > 40: 
                            if region_has_symbol(comp_aligned, match["bbox"]):
                                base_draw_actions.append((base_sym["bbox"], color_misplaced, "Misplaced", False))
                                child_draw_actions.append((match["bbox"], color_misplaced, "Misplaced", False))
                else:
                    base_draw_actions.append((base_sym["bbox"], color_deleted, "Deleted", False))

            for b_box, c_box in modified_image_boxes:
                base_draw_actions.append((b_box, color_modified, "Modified", False))
                child_draw_actions.append((c_box, color_modified, "Modified", False))
                
            for box in added_img_boxes + added_bc_boxes:
                child_draw_actions.append((box, color_added, "Added", False))
                
            for box in deleted_img_boxes + deleted_bc_boxes:
                base_draw_actions.append((box, color_deleted, "Deleted", False))

            for d in comp_symbols_raw:
                if d["class"] not in [b["class"] for b in base_symbols_raw]:
                    child_draw_actions.append((d["bbox"], color_added, "Added", False))

            base_img_out = base_processed.copy()
            child_img_out = comp_aligned.copy()
            draw_b = ImageDraw.Draw(base_img_out)
            draw_c = ImageDraw.Draw(child_img_out)

            try:
                # Use a larger font if available, fallback to default
                from PIL import ImageFont
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = None

            def apply_draws(draw_obj, actions):
                for bbox, color, label, is_xywh in actions:
                    # Robust check: Must be a sequence of exactly 4 coordinates
                    if not bbox or not hasattr(bbox, '__len__') or len(bbox) != 4:
                        continue
                        
                    try:
                        if is_xywh:
                            x, y, w, h = map(int, bbox)
                            x1, y1, x2, y2 = x, y, x+w, y+h
                        else:
                            x1, y1, x2, y2 = map(int, bbox)
                    except (TypeError, ValueError):
                        # Skip if there's any issue converting coordinates to integers
                        continue
                    draw_obj.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    # Add background for text to make it readable
                    text_bbox = draw_obj.textbbox((x1, max(0, y1-25)), label, font=font) if font else draw_obj.textbbox((x1, max(0, y1-15)), label)
                    draw_obj.rectangle(text_bbox, fill=(255, 255, 255))
                    draw_obj.text((x1, max(0, y1-25) if font else max(0, y1-15)), label, fill=color, font=font)

            apply_draws(draw_b, base_draw_actions)
            apply_draws(draw_c, child_draw_actions)

            import base64
            def img_to_base64(img):
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
            
            b64_base = img_to_base64(base_img_out)
            b64_child = img_to_base64(child_img_out)
            # ---- END BOUNDING BOXES ----

            final_results.append({
                "filename": child_file.filename,
                "discrepancies": discrepancy_report,
                "features_extracted": {
                    "base": base_features_records,
                    "child": comp_features_records
                },
                "annotated_base_image": b64_base,
                "annotated_child_image": b64_child
            })

        return JSONResponse(content={"results": final_results})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
