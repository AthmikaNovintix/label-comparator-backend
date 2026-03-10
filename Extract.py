import pandas as pd
import numpy as np
import cv2
import os
import zxingcpp
import pytesseract
import sys
import os

# Intelligently find Tesseract-OCR on Windows if not in PATH
if sys.platform == "win32":
    tess_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
    ]
    for p in tess_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break

import re
import traceback
from PIL import Image
import gc

# --- Prevent PyTorch/Tesseract from using too much RAM ---
import torch
torch.set_num_threads(1) 

from detect import run_detection_pil

# ---------------------------------------------------------
# ADVANCED BARCODE SCANNER LOGIC (From agentic_scanner.py)
# ---------------------------------------------------------
def _zxing_decode(image):
    """Internal helper to run ZXing with advanced parameters"""
    found = []
    try:
        zx_results = zxingcpp.read_barcodes(
            image, 
            try_rotate=True, 
            try_invert=True,
            try_downscale=True
        )
        for res in zx_results:
            # ZXing position normally returns a 'position' object with corner points
            try:
                p = res.position
                x1 = min(p.top_left.x, p.bottom_left.x)
                y1 = min(p.top_left.y, p.top_right.y)
                x2 = max(p.top_right.x, p.bottom_right.x)
                y2 = max(p.bottom_left.y, p.bottom_right.y)
                bbox = [int(x1), int(y1), int(x2), int(y2)]
            except:
                bbox = None
                
            found.append({"data": res.text, "type": res.format.name, "bbox": bbox})
    except Exception as e:
        print(f"Error in zxing decode: {e}")
    return found

def extract_barcodes(image):
    """Master Barcode Extractor with Agentic Fallback & Preprocessing"""
    img_np = np.array(image)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    results = []

    # 1. Standard Decode
    results.extend(_zxing_decode(img_np))

    # 2. Advanced Preprocessing Passes for tricky barcodes
    if not any(r['type'] in ['Code128', 'Code39', 'EAN8', 'EAN13', 'UPCA'] for r in results):
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        # 2a. Multi-scale
        results.extend(_zxing_decode(cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)))
        results.extend(_zxing_decode(cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)))
        
        # 2b. Otsu Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.extend(_zxing_decode(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))
        
        # 2c. Morphological Closing (helps repair broken bars)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        results.extend(_zxing_decode(cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)))
        
        # 2d. Horizontal Blur to connect bars vertically (good for 1D barcodes)
        blurred = cv2.GaussianBlur(gray, (1, 15), 0)
        _, thresh_blur = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.extend(_zxing_decode(cv2.cvtColor(thresh_blur, cv2.COLOR_GRAY2BGR)))

        # 2e. Scanline Averaging for Multiple Patches (Top, Mid, Bottom)
        h, w = img_np.shape[:2]
        patches = [
            gray[0:int(h*0.3), :],                  # Top
            gray[int(h*0.35):int(h*0.65), :],       # Middle
            gray[int(h*0.7):h, :]                   # Bottom
        ]
        
        for p in patches:
            if p.shape[0] < 10: continue
            avg_line = np.mean(p, axis=0).astype(np.uint8)
            scan_img = np.tile(avg_line, (200, 1))
            padded = cv2.copyMakeBorder(scan_img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
            processed = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
            processed = cv2.resize(processed, None, fx=2.0, fy=1.0, interpolation=cv2.INTER_CUBIC)
            results.extend(_zxing_decode(processed))

    # 4. De-duplicate results while preserving boxes
    unique_barcodes = []
    seen = set()
    for res in results:
        data = res['data']
        if data not in seen:
            seen.add(data)
            unique_barcodes.append({"data": data, "bbox": res.get("bbox")})
            
    return unique_barcodes

# ---------------------------------------------------------
# EXISTING LOGO, TEXT, AND MASTER EXTRACTION LOGIC
# ---------------------------------------------------------
def detect_logos(image, logo_folder="logos"):
    """Compares the label against a folder of known logos using SIFT + RANSAC Geometry Verification"""
    if not os.path.exists(logo_folder):
        return []

    label_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(label_gray, None)
    
    if des1 is None or len(des1) < 2:
        return []
    
    detected_logos = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    for logo_file in os.listdir(logo_folder):
        if not logo_file.lower().endswith(valid_extensions):
            continue
            
        logo_path = os.path.join(logo_folder, logo_file)
        
        try:
            logo_img = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
            if logo_img is None: continue
                
            kp2, des2 = sift.detectAndCompute(logo_img, None)
            if des2 is None or len(des2) < 2: continue
            
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des2, des1, k=2)
            
            good_matches = []
            for match_group in matches:
                if len(match_group) == 2:
                    m, n = match_group
                    # Relaxed distance threshold for text-heavy logos
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # ---------------------------------------------------------
            # THE FIX: RANSAC GEOMETRY VERIFICATION
            # ---------------------------------------------------------
            # Threshold lowered to 10 to catch small/text logos safely
            if len(good_matches) >= 10: 
                src_pts = np.float32([ kp2[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
                
                # RANSAC ensures the matching points actually form the physical shape of the logo
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if mask is not None:
                    inliers = np.sum(mask)
                    # If points form a valid geometric shape, it's a confirmed logo
                    if inliers >= 10 and inliers >= (0.05 * len(kp2)):
                        h, w = logo_img.shape[:2]
                        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                        dst = cv2.perspectiveTransform(pts, M)
                        x, y, bw, bh = cv2.boundingRect(dst)
                        detected_logos.append({"name": logo_file.rsplit('.', 1)[0], "bbox": [x, y, x+bw, y+bh]})
                        
        except Exception as e:
            print(f"Skipping {logo_file} due to error:")
            import traceback
            traceback.print_exc()
            continue
            
    # Return unique values only to prevent duplicates based on name
    unique_logos = []
    seen = set()
    for l in detected_logos:
        if l["name"] not in seen:
            seen.add(l["name"])
            unique_logos.append(l)
    return unique_logos 

def extract_all_features(image, precomputed_symbols, logo_folder="logos"):
    """Master function to extract Text, Barcodes, Logos, and append Symbols"""
    features = []

    # 1. Advanced Barcode Extraction (Includes Agentic Scanner Logic)
    barcodes_raw = extract_barcodes(image)
    barcode_values = [b["data"] for b in barcodes_raw]
    for bc in barcodes_raw:
        features.append({"Type": "Barcode", "Value": bc["data"], "Box": bc["bbox"]})

    # 2. Logo / Image Extraction (Now powered by RANSAC)
    logos = detect_logos(image, logo_folder)
    for logo in logos:
        features.append({"Type": "Image", "Value": f"Image - {logo['name']}", "Box": logo["bbox"]})

    # 3. Append Precomputed YOLO Symbols
    for sym in precomputed_symbols:
        features.append({"Type": "Symbol", "Value": sym["class"], "Box": sym["bbox"]})

    # 4. Ultra-lightweight Text Extraction (Tesseract)
    np_img = np.array(image)
    if len(np_img.shape) == 3 and np_img.shape[2] == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img
        
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # Get HOCR or data to have boxes for OCR-detected barcodes
    ocr_data = pytesseract.image_to_data(gray, lang='eng+fra+deu', output_type=pytesseract.Output.DICT)
    ocr_text = pytesseract.image_to_string(gray, lang='eng+fra+deu')
    
    # Convert barcodes to sets of stripped strings for robust duplicate checking
    barcode_compact = [re.sub(r'\s+', '', bc).lower() for bc in barcode_values]
    
    for line in ocr_text.split('\n'):
        clean_text = re.sub(r'[|><_~=«»"*;]', '', line).strip()
        compact_text = re.sub(r'\s+', '', clean_text).lower()
        
        if len(clean_text) > 2 and any(c.isalnum() for c in clean_text):
            # Check if this text is basically an extracted barcode
            is_barcode_already_found = any(compact_text in bc or bc in compact_text for bc in barcode_compact if len(bc) > 4)
            
            # Heuristic for undetected 1D barcodes that OCR picks up:
            looks_like_raw_barcode = (
                len(compact_text) > 10 and 
                compact_text.isalnum() and
                clean_text.isupper() and 
                not any(c.islower() for c in clean_text) and
                sum(c.isdigit() for c in clean_text) > 2
            )

            if is_barcode_already_found:
                continue  # Skip, it's already a barcode feature
            elif looks_like_raw_barcode:
                # Find coarse box from OCR data for this text
                bbox = None
                try:
                    # Search for the clean_text in ocr_data words
                    # Tesseract data is word-by-word, so we look for matching strings
                    for i, word in enumerate(ocr_data['text']):
                        if compact_text in re.sub(r'\s+', '', word).lower():
                            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                            # Scale back down (we resized by 1.5x)
                            bbox = [int(x/1.5), int(y/1.5), int((x+w)/1.5), int((y+h)/1.5)]
                            break
                except:
                    pass
                
                barcode_values.append(clean_text)
                barcode_compact.append(compact_text)
                features.append({"Type": "Barcode", "Value": clean_text, "Box": bbox})
            else:
                features.append({"Type": "Text", "Value": clean_text, "Box": None})

    return pd.DataFrame(features)
