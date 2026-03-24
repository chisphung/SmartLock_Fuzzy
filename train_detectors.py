import os
import cv2
import time
import argparse
import random
import numpy as np
try:
    import joblib
except ImportError:
    import pickle as joblib

try:
    import dlib
except ImportError:
    dlib = None

try:
    from skimage.feature import hog
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

WIDER_TRAIN_DIR = "/home/chisphung/CE224.Q11_People_Counting/WIDER_train/images"
WIDER_TRAIN_TXT = "/home/chisphung/CE224.Q11_People_Counting/wider_face_split/wider_face_train_bbx_gt.txt"
OUTPUT_DIR = "custom_models"

def parse_wider_annotations(txt_file, base_dir, limit=None):
    """Parses WIDER FACE ground truths into a dictionary {filepath: [bboxes]}"""
    dataset = {}
    if not os.path.exists(txt_file):
        print(f"Error: Could not find {txt_file}")
        return dataset
        
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    count = 0
    while i < len(lines):
        if limit and count >= limit:
            break
            
        filename = lines[i].strip()
        if not filename.endswith('.jpg'):
            i += 1
            continue
            
        i += 1
        try:
            num_boxes = int(lines[i].strip())
        except ValueError:
            num_boxes = 0
            
        i += 1
        bboxes = []
        for _ in range(num_boxes):
            if i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 4:
                    x1, y1, w, h = map(int, parts[:4])
                    if w >= 40 and h >= 40: # Filter out faces that are too tiny for HOG training
                        bboxes.append([x1, y1, w, h])
            i += 1
            
        full_path = os.path.join(base_dir, filename)
        if len(bboxes) > 0 and os.path.exists(full_path):
            dataset[full_path] = bboxes
            count += 1
            
    return dataset

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0]+boxA[2], boxB[0]+boxB[2]), min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea, boxBArea = boxA[2]*boxA[3], boxB[2]*boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ---- Approach A: Scikit Learn (From Scratch HOG+SVM) ----
def calculate_hog(img):
    # Using 64x64 standard resize for HOG detection patches
    resized = cv2.resize(img, (64, 64))
    fd = hog(resized, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), visualize=False, transform_sqrt=True)
    return fd

def train_approach_a(val_data):
    if not SKLEARN_AVAILABLE:
        print("ERROR: Please install scikit-learn and scikit-image: pip install scikit-learn scikit-image")
        return

    print("Extracting Positive and Negative patches...")
    positive_features = []
    negative_features = []
    
    for idx, (img_path, bboxes) in enumerate(val_data.items()):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # 1. Positives
        for (x, y, w, h) in bboxes:
            crop = img[y:y+h, x:x+w]
            positive_features.append(calculate_hog(crop))
            
        # 2. Negatives (Random Background Mining)
        # Sample patches from background
        img_h, img_w = img.shape
        num_neg_per_img = 5
        for _ in range(num_neg_per_img):
            nw, nh = 64, 64
            if img_w <= nw or img_h <= nh: continue
            nx = random.randint(0, img_w - nw)
            ny = random.randint(0, img_h - nh)
            nbox = [nx, ny, nw, nh]
            
            # Ensure it doesn't overlap strongly with a face
            max_iou = max([iou(nbox, fb) for fb in bboxes] + [0])
            if max_iou < 0.1:
                crop = img[ny:ny+nh, nx:nx+nw]
                negative_features.append(calculate_hog(crop))
                
        if idx % 500 == 0 and idx > 0:
            print(f"  Processed {idx} images... (Pos: {len(positive_features)}, Neg: {len(negative_features)})")
            
    # Prepare Arrays
    X = np.vstack(positive_features + negative_features)
    y = np.hstack([np.ones(len(positive_features)), np.zeros(len(negative_features))])
    
    print(f"\nTraining Linear SVM on {len(X)} samples... (This may take a minute)")
    start_t = time.time()
    model = LinearSVC(C=0.01, max_iter=2000, class_weight='balanced')
    model.fit(X, y)
    print(f"Training Complete in {time.time() - start_t:.2f}s")
    
    # Save Model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'sklearn_hog_face_svm.pkl')
    joblib.dump(model, out_path)
    print(f"Model saved to {out_path}")
    print("NOTE: For a true detector, a Sliding Window & Hard Negative Mining loop would be executed next using this base SVM.")

# ---- Approach B: Dlib Max-Margin Object Detector ----
def write_dlib_xml(val_data, out_xml):
    with open(out_xml, 'w') as f:
        f.write("<?xml version='1.0' encoding='ISO-8859-1'?>\n")
        f.write("<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n")
        f.write("<dataset>\n<name>WIDER_Subset_Dlib</name>\n<images>\n")
        for img_path, bboxes in val_data.items():
            f.write(f"  <image file='{img_path}'>\n")
            for (x, y, w, h) in bboxes:
                f.write(f"    <box top='{y}' left='{x}' width='{w}' height='{h}'/>\n")
            f.write("  </image>\n")
        f.write("</images>\n</dataset>\n")

def train_approach_b(val_data):
    if dlib is None:
        print("ERROR: Dlib not installed. Cannot train Approach B. Please install dlib (requires cmake).")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xml_path = os.path.join(OUTPUT_DIR, "dlib_training.xml")
    svm_path = os.path.join(OUTPUT_DIR, "dlib_face_detector.svm")
    
    print(f"Generating Dlib XML format to {xml_path}...")
    write_dlib_xml(val_data, xml_path)
    
    print("\nTraining Dlib Structural SVM (This performs automatic internal hard-negative mining)...")
    options = dlib.simple_object_detector_training_options()
    options.add_left_right_image_flips = True
    options.C = 5
    options.num_threads = 4  # Adjust based on CPU
    options.be_verbose = True
    
    start_t = time.time()
    try:
        dlib.train_simple_object_detector(xml_path, svm_path, options)
        print(f"Training Complete in {time.time() - start_t:.2f}s")
        print(f"Model saved to {svm_path}")
        
        print("\nEvaluating trained model on its own training set:")
        metrics = dlib.test_simple_object_detector(xml_path, svm_path)
        print("Dlib training set metrics:", metrics)
    except Exception as e:
        print(f"Dlib training failed: {e}")
        print("This usually happens if the crops are too small or varied in aspect ratio for Dlib's rigid HOG window. ")
        print("We may need to clean extreme aspect ratios from the WIDER dataset first.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--approach", choices=["A", "B", "ALL"], default="ALL", help="Which training approach to run")
    p.add_argument("--limit", type=int, default=500, help="Number of training images to subset. Full dataset takes massive RAM.")
    args = p.parse_args()
    
    print(f"Loading WIDER FACE training annotations (Limit: {args.limit} images)...")
    train_data = parse_wider_annotations(WIDER_TRAIN_TXT, WIDER_TRAIN_DIR, limit=args.limit)
    print(f"Loaded {len(train_data)} workable images containing faces.")
    
    if len(train_data) == 0:
        print("No valid training images found. Check WIDER paths.")
        return

    if args.approach in ["A", "ALL"]:
        print("\n" + "="*50)
        print("STARTING APPROACH A: Scikit Learn (Base HOG+SVM)")
        print("="*50)
        train_approach_a(train_data)

    if args.approach in ["B", "ALL"]:
        print("\n" + "="*50)
        print("STARTING APPROACH B: Dlib Structural SVM")
        print("="*50)
        train_approach_b(train_data)

if __name__ == "__main__":
    main()
