import os
import cv2
import time
import numpy as np

# Configuration
WIDER_TRAIN_DIR = "/home/chisphung/CE224.Q11_People_Counting/WIDER_train/images"
WIDER_VAL_DIR = "/home/chisphung/CE224.Q11_People_Counting/WIDER_val/images"
WIDER_TRAIN_TXT = "/home/chisphung/CE224.Q11_People_Counting/wider_face_split/wider_face_train_bbx_gt.txt"
WIDER_VAL_TXT = "/home/chisphung/CE224.Q11_People_Counting/wider_face_split/wider_face_val_bbx_gt.txt"

# ---- 1. Data Loader ----
def parse_wider_annotations(txt_file, base_dir):
    """Parses WIDER FACE bounding box ground truths."""
    dataset = {}
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        filename = lines[i].strip()
        if not filename.endswith('.jpg'):
            i += 1
            continue
        
        i += 1
        # Handle edge cases where num_boxes might not be castable or the format is slightly off
        try:
            num_boxes = int(lines[i].strip())
        except ValueError:
            i += 1
            num_boxes = 0
            
        i += 1
        bboxes = []
        for _ in range(num_boxes):
            if i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 4:
                    x1, y1, w, h = map(int, parts[:4])
                    if w > 0 and h > 0: # Only keep valid boxes
                        bboxes.append([x1, y1, w, h])
            i += 1
            
        full_path = os.path.join(base_dir, filename)
        if os.path.exists(full_path):
            dataset[full_path] = bboxes
            
    return dataset

# ---- 2. Face Detection Metrics ----
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# ---- 3. Detectors ----
class HaarDetector:
    def __init__(self):
        import os
        cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
    def detect(self, gray_img):
        faces = self.cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        return faces

class HOGDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # Note: Default OpenCV HOG is for full human body. 
        # For faces in Python without dlib, we rely on Haar or a custom SVM. 
        # Here we mock Face HOG using the built-in pedestrian or a Haar fallback if strictly faces.
        # Since standard OpenCV Python lacks a built-in pre-trained HOG *Face* detector,
        # we demonstrate the class framework but use Haar parameters to simulate the API structure,
        # or require dlib explicitly.
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.use_dlib = True
        except ImportError:
            print("Warning: Dlib not found. Standard OpenCV HOG is for bodies. Falling back to Haar for Face HOG simulation.")
            self.use_dlib = False
            self.fallback = HaarDetector()

    def detect(self, gray_img):
        if self.use_dlib:
            faces = self.detector(gray_img, 1)
            bboxes = []
            for face in faces:
                bboxes.append([face.left(), face.top(), face.width(), face.height()])
            return bboxes
        else:
            return self.fallback.detect(gray_img)

# ---- 4. Recognizers (Pseudo Identity Generator) ----
# Note: WIDER does not have actual ID labels. We create pseudo IDs by taking WIDER_val face crops.
def prepare_pseudo_recognition_dataset(val_data, num_identities=10, samples_per_id=5):
    """Crops faces from WIDER_val and applies augmentation to simulate a recognition dataset."""
    identities = []
    labels = []
    
    current_id = 0
    for img_path, bboxes in val_data.items():
        if current_id >= num_identities:
            break
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        for (x, y, w, h) in bboxes:
            if w < 50 or h < 50: continue # Need minimum size for recognition
            face_crop = img[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (100, 100))
            
            # Create augmented samples for training
            for _ in range(samples_per_id):
                # Random slight translation/brightness for pseudo-variance
                M = np.float32([[1, 0, np.random.randint(-5, 5)], [0, 1, np.random.randint(-5, 5)]])
                aug_face = cv2.warpAffine(face_crop, M, (100, 100))
                identities.append(aug_face)
                labels.append(current_id)
            current_id += 1
            if current_id >= num_identities:
                break
    return identities, np.array(labels)

class LBPHRecognizer:
    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer_create()
    def train(self, faces, labels):
        self.model.train(faces, labels)
    def predict(self, face):
        label, conf = self.model.predict(face)
        return label, conf

class EigenRecognizer:
    def __init__(self):
        self.model = cv2.face.EigenFaceRecognizer_create()
    def train(self, faces, labels):
        self.model.train(faces, labels)
    def predict(self, face):
        label, conf = self.model.predict(face)
        return label, conf

# ---- 5. Ablation Pipeline ----
def evaluate_detection(detector, val_data, limit=100):
    total_fps = 0
    total_iou = 0
    hits = 0
    total_gt = 0
    
    count = 0
    for img_path, gt_bboxes in val_data.items():
        if count >= limit: break
        count += 1
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        start_time = time.time()
        pred_bboxes = detector.detect(img)
        latency = time.time() - start_time
        total_fps += (1.0 / (latency + 1e-6))
        
        total_gt += len(gt_bboxes)
        
        # Simple Greedy IoU Matcher
        matched = set()
        for p_box in pred_bboxes:
            best_iou = 0
            best_gt = -1
            for j, g_box in enumerate(gt_bboxes):
                if j in matched: continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = j
            if best_iou > 0.4:
                hits += 1
                total_iou += best_iou
                matched.add(best_gt)
                
    avg_fps = total_fps / count if count > 0 else 0
    recall = hits / float(total_gt) if total_gt > 0 else 0
    avg_iou = total_iou / float(hits) if hits > 0 else 0
    return avg_fps, recall, avg_iou

def build_and_evaluate_pipeline():
    print("Loading WIDER_val annotations...")
    val_data = parse_wider_annotations(WIDER_VAL_TXT, WIDER_VAL_DIR)
    print(f"Loaded {len(val_data)} validation images.")
    
    print("\n--- PHASE 1: Detection Ablation (Haar vs HOG) ---")
    haar = HaarDetector()
    hog = HOGDetector()
    
    print("Evaluating Haar...")
    haar_fps, haar_rec, haar_iou = evaluate_detection(haar, val_data, limit=50)
    print(f"Haar   -> FPS: {haar_fps:.1f} | Recall: {haar_rec:.2f} | Avg IoU: {haar_iou:.2f}")
    
    print("Evaluating HOG...")
    hog_fps, hog_rec, hog_iou = evaluate_detection(hog, val_data, limit=50)
    print(f"HOG    -> FPS: {hog_fps:.1f} | Recall: {hog_rec:.2f} | Avg IoU: {hog_iou:.2f}")
    
    print("\n--- PHASE 2: Recognition Ablation (LBPH vs Eigenfaces) ---")
    try:
        import cv2.face
    except AttributeError:
        print("ERROR: opencv-contrib-python is required for cv2.face. Install via: pip install opencv-contrib-python")
        return
        
    print("Preparing pseudo-recognition dataset from valid face crops...")
    faces, labels = prepare_pseudo_recognition_dataset(val_data, num_identities=15, samples_per_id=5)
    
    if len(faces) == 0:
        print("Could not extract enough valid faces for recognition testing.")
        return
        
    # Split into mock train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)
    
    classifiers = {
        "LBPH": LBPHRecognizer(),
        "Eigenfaces": EigenRecognizer()
    }
    
    for name, clf in classifiers.items():
        start_train = time.time()
        clf.train(X_train, y_train)
        train_time = time.time() - start_train
        
        correct = 0
        infer_times = []
        for i in range(len(X_test)):
            start_inf = time.time()
            pred, conf = clf.predict(X_test[i])
            infer_times.append(time.time() - start_inf)
            if pred == y_test[i]:
                correct += 1
                
        acc = correct / len(X_test)
        avg_infer = np.mean(infer_times) * 1000 # ms
        print(f"{name:10s} -> Accuracy: {acc*100:.1f}% | Train Time: {train_time:.3f}s | Infer Time: {avg_infer:.2f}ms/face")

if __name__ == "__main__":
    build_and_evaluate_pipeline()
