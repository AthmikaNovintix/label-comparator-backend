from ultralytics import YOLO
import cv2
import math
import tempfile
import gc

# -----------------------------
# Run Detection (For Streamlit/PIL)
# -----------------------------
def run_detection_pil(pil_image, label_name="Document"):
    """Handles PIL images using a Load-and-Dump memory approach with detailed logging"""
    print(f"\n{'='*60}")
    print(f"🤖 YOLO AI SYMBOL DETECTION STARTING")
    print(f"{'='*60}")
    
    # 1. Load models into RAM
    print("[1/3] Loading neural networks into memory...")
    try:
        model16 = YOLO("16sym_models/best.pt")
        model4 = YOLO("4sym_models/best.pt")
    except Exception as e:
        print(f"❌ ERROR LOADING MODELS: {e}")
        return []
    
    # 2. Process the image
    print(f"[2/3] Scanning image pixels...")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_image.save(tmp.name)
        # verbose=False hides the messy default Ultralytics logs so our clean logs shine
        results16 = model16(tmp.name, conf=0.3, verbose=False)[0]
        results4 = model4(tmp.name, conf=0.3, verbose=False)[0]
        
    detections = []
    print("\n--- EXTRACTED SYMBOLS ---")
    
    for box in results16.boxes:
        cls = model16.names[int(box.cls)]
        bbox = box.xyxy[0].tolist()
        conf = float(box.conf)
        print(f"  ✓ {cls:<20} | Conf: {conf:.2f} | Coords: {[int(x) for x in bbox]}")
        detections.append({"class": cls, "bbox": bbox, "label": "Symbol"})
        
    for box in results4.boxes:
        cls = model4.names[int(box.cls)]
        bbox = box.xyxy[0].tolist()
        conf = float(box.conf)
        print(f"  ✓ {cls:<20} | Conf: {conf:.2f} | Coords: {[int(x) for x in bbox]}")
        detections.append({"class": cls, "bbox": bbox, "label": "Symbol"})
        
    if not detections:
        print("  [!] No symbols detected in this document.")
    else:
        print(f"  Total Symbols Found: {len(detections)}")
        
    # 3. CRITICAL: Delete models and force RAM cleanup
    print("\n[3/3] Flushing models from RAM...")
    del model16
    del model4
    gc.collect()
    
    print(f"{'='*60}\n")
    return detections

# -----------------------------
# Utility / Math Logic
# -----------------------------
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def compare_labels(base_det, edited_det, threshold=40):
    added = []
    removed = []
    misplaced = []
    
    print(f"\n🔍 RUNNING SYMBOL DISCREPANCY ENGINE")
    print("-" * 40)
    
    base_classes = [d["class"] for d in base_det]
    edited_classes = [d["class"] for d in edited_det]
    
    for d in edited_det:
        if d["class"] not in base_classes:
            d = d.copy()
            d["label"] = "Added"
            added.append(d)
            print(f"  [+] ADDED:     {d['class']}")
            
    for d in base_det:
        if d["class"] not in edited_classes:
            d = d.copy()
            d["label"] = "Removed"
            removed.append(d)
            print(f"  [-] DELETED:   {d['class']}")
            
    for b in base_det:
        for e in edited_det:
            if b["class"] == e["class"]:
                c1 = get_center(b["bbox"])
                c2 = get_center(e["bbox"])
                dist = math.dist(c1, c2)
                if dist > threshold:
                    e = e.copy()
                    e["label"] = "Repositioned"
                    misplaced.append(e)
                    print(f"  [~] MISPLACED: {e['class']} (Shifted {int(dist)} pixels)")
                    
    print("-" * 40)
    print(f"  Summary: {len(added)} Added | {len(removed)} Deleted | {len(misplaced)} Misplaced\n")
    
    return added, removed, misplaced