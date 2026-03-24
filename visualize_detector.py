import os
import cv2
import dlib

WIDER_VAL_DIR = "/home/chisphung/CE224.Q11_People_Counting/WIDER_val/images"
WIDER_VAL_TXT = "/home/chisphung/CE224.Q11_People_Counting/wider_face_split/wider_face_val_bbx_gt.txt"
MODEL_PATH = "/home/chisphung/CE224.Q11_People_Counting/custom_models/dlib_face_detector.svm"
OUT_DIR = "/home/chisphung/CE224.Q11_People_Counting/visualizations"

def parse_wider_annotations(txt_file, base_dir, limit=None):
    dataset = {}
    if not os.path.exists(txt_file): return dataset
    with open(txt_file, 'r') as f: lines = f.readlines()
    i, count = 0, 0
    while i < len(lines):
        if limit and count >= limit: break
        filename = lines[i].strip()
        if not filename.endswith('.jpg'): i += 1; continue
        i += 1
        try: num_boxes = int(lines[i].strip())
        except ValueError: num_boxes = 0
        i += 1
        bboxes = []
        for _ in range(num_boxes):
            if i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 4:
                    x1, y1, w, h = map(int, parts[:4])
                    if w >= 20 and h >= 20: bboxes.append([x1, y1, w, h])
            i += 1
        full_path = os.path.join(base_dir, filename)
        if len(bboxes) > 0 and os.path.exists(full_path):
            dataset[full_path] = bboxes
            count += 1
    return dataset

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}")
        return
        
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Loading test data...")
    val_data = parse_wider_annotations(WIDER_VAL_TXT, WIDER_VAL_DIR, limit=10)
    
    print("Loading Dlib model...")
    detector = dlib.simple_object_detector(MODEL_PATH)
    
    print(f"Running inference and saving visualizations to {OUT_DIR}...")
    for idx, (img_path, gt_bboxes) in enumerate(val_data.items()):
        img = cv2.imread(img_path)
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ground Truths in GREEN
        for (x, y, w, h) in gt_bboxes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "GT", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # Predictions in RED
        preds = detector(gray)
        for p in preds:
            cv2.rectangle(img, (p.left(), p.top()), (p.right(), p.bottom()), (0, 0, 255), 3)
            cv2.putText(img, "Pred", (p.left(), p.top()-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        out_path = os.path.join(OUT_DIR, f"result_{idx}.jpg")
        cv2.imwrite(out_path, img)
        print(f"Saved {out_path} (Found {len(preds)} faces, GT had {len(gt_bboxes)})")

    print(f"\nDone! Open the files in {OUT_DIR} to see your trained detector at work.")

if __name__ == "__main__":
    main()
