import os
import cv2
import time
import argparse
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
    from skimage.transform import pyramid_gaussian
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


WIDER_VAL_DIR = "/home/chisphung/CE224.Q11_People_Counting/WIDER_val/images"
WIDER_VAL_TXT = "/home/chisphung/CE224.Q11_People_Counting/wider_face_split/wider_face_val_bbx_gt.txt"
MODELS_DIR = "custom_models"

# 1. Parsing and Metrics identical to previous scripts
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

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0]+boxA[2], boxB[0]+boxB[2]), min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea, boxBArea = boxA[2]*boxA[3], boxB[2]*boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0: return []
    if boxes.dtype.kind == "i": boxes = boxes.astype("float")
    pick = []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int").tolist()

# 2. Custom Sklearn SVM Detector
class CustomSVMDetector:
    def __init__(self, model_path):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-image is not installed")
        self.model = joblib.load(model_path)
        self.window_size = (64, 64)
        
    def sliding_window(self, image, stepSize, windowSize):
        for y in range(0, image.shape[0] - windowSize[1], stepSize):
            for x in range(0, image.shape[1] - windowSize[0], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    def detect(self, img, threshold=0.5):
        # NOTE: Native python sliding window is a pedagogical tool. In production, 
        # C++ wrappers or Dlib are necessary for speed.
        rectifications = []
        
        # We perform one basic scale for demonstration due to python execution constraints
        # Real pyramid would loop: for scale, resized in enumerate(pyramid_gaussian(img, downscale=1.5)):
        for (x, y, window) in self.sliding_window(img, stepSize=16, windowSize=self.window_size):
            if window.shape[0] != self.window_size[1] or window.shape[1] != self.window_size[0]: continue
            features = hog(window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, transform_sqrt=True)
            # Use decision_function to get confidence instead of predict()
            conf = self.model.decision_function(features.reshape(1, -1))[0]
            if conf > threshold:
                rectifications.append([x, y, x + self.window_size[0], y + self.window_size[1]])
        
        if len(rectifications) > 0:
            rects_np = np.array(rectifications)
            final_boxes = non_max_suppression_fast(rects_np, 0.3)
            # Convert x1, y1, x2, y2 -> x, y, w, h
            return [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in final_boxes]
        return []

# 3. Custom Dlib Detector
class CustomDlibDetector:
    def __init__(self, model_path):
        if not dlib:
            raise ImportError("Dlib is needed")
        self.detector = dlib.simple_object_detector(model_path)
            
    def detect(self, img):
        dets = self.detector(img)
        bboxes = []
        for d in dets:
            bboxes.append([d.left(), d.top(), d.width(), d.height()])
        return bboxes

# 4. Evaluator
def evaluate_model(name, detector_func, val_data, limit=50):
    total_fps = 0
    hits = 0
    total_gt = 0
    count = 0
    
    for img_path, gt_bboxes in val_data.items():
        if count >= limit: break
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        count += 1
        
        start = time.time()
        pred_bboxes = detector_func(img)
        latency = time.time() - start
        
        total_fps += (1.0 / (latency + 1e-6))
        total_gt += len(gt_bboxes)
        
        # Simple IoU Matcher
        matched = set()
        for p_box in pred_bboxes:
            best_iou = 0; best_gt = -1
            for j, g_box in enumerate(gt_bboxes):
                if j in matched: continue
                val = iou(p_box, g_box)
                if val > best_iou: best_iou = val; best_gt = j
            if best_iou > 0.3: # Relaxed slightly since custom models might box tighter
                hits += 1
                matched.add(best_gt)

    avg_fps = total_fps / count if count > 0 else 0
    recall = hits / float(total_gt) if total_gt > 0 else 0
    print(f"[{name}] -> Average FPS: {avg_fps:.2f} | Ground Truth Faces Found (Recall): {recall*100:.1f}%")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50, help="Number of images to evaluate")
    args = p.parse_args()

    print("Loading valid WIDER face validation ground truths...")
    val_data = parse_wider_annotations(WIDER_VAL_TXT, WIDER_VAL_DIR, limit=args.limit)
    print(f"Prepared {len(val_data)} validation images.")

    # 1. Approach A Evaluation
    svm_path = os.path.join(MODELS_DIR, 'sklearn_hog_face_svm.pkl')
    if os.path.exists(svm_path) and SKLEARN_AVAILABLE:
        print("\nEvaluating Custom Sklearn HOG+SVM...")
        detectorA = CustomSVMDetector(svm_path)
        evaluate_model("Sklearn SVM", detectorA.detect, val_data, limit=args.limit)
    else:
        print(f"\nSkipping Sklearn SVM. Model not found at {svm_path} or dependencies missing.")

    # 2. Approach B Evaluation
    dlib_path = os.path.join(MODELS_DIR, 'dlib_face_detector.svm')
    if os.path.exists(dlib_path) and dlib:
        print("\nEvaluating Custom Dlib Structural SVM...")
        detectorB = CustomDlibDetector(dlib_path)
        evaluate_model("Dlib SVM", detectorB.detect, val_data, limit=args.limit)
    else:
        print(f"\nSkipping Dlib SVM. Model not found at {dlib_path} or dlib not installed.")

if __name__ == "__main__":
    main()
