import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================
BASE_PATH = r"C:\Users\nared\Desktop\Task_2\KITTI_Selection"
IMAGE_PATH = os.path.join(BASE_PATH, "images")
CALIB_PATH = os.path.join(BASE_PATH, "calib")
LABEL_PATH = os.path.join(BASE_PATH, "labels")

OUTPUT_PATH = os.path.join(BASE_PATH, "Output")
IMG_OUT_PATH = os.path.join(OUTPUT_PATH, "images")
PLOT_OUT_PATH = os.path.join(OUTPUT_PATH, "plots")
LOG_OUT_PATH = os.path.join(OUTPUT_PATH, "logs")

os.makedirs(IMG_OUT_PATH, exist_ok=True)
os.makedirs(PLOT_OUT_PATH, exist_ok=True)
os.makedirs(LOG_OUT_PATH, exist_ok=True)

# ============================================================
# CONSTANTS
# ============================================================
CAMERA_HEIGHT = 1.65  # meters
IOU_THRESHOLD = 0.5

# ============================================================
# LOAD YOLO (PRE-TRAINED)
# ============================================================
model = YOLO("yolov8n.pt")

# ============================================================
# LOAD CALIBRATION
# ============================================================
def load_calibration(file):
    return np.loadtxt(file)

# ============================================================
# LOAD LABELS (GROUND TRUTH)
# ============================================================
def load_labels(file):
    objects = []
    with open(file, "r") as f:
        for line in f:
            p = line.strip().split()
            cls = p[0]
            xmin, ymin, xmax, ymax = map(float, p[1:5])
            dist = float(p[5])
            objects.append((cls, (xmin, ymin, xmax, ymax), dist))
    return objects

# ============================================================
# IOU
# ============================================================
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

# ============================================================
# DEPTH ESTIMATION (GROUND-PLANE GEOMETRY)
# ============================================================
def estimate_depth(K, bbox):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    xmin, ymin, xmax, ymax = bbox
    u = (xmin + xmax) / 2
    v = ymax

    x = (u - cx) / fx
    y = -(v - cy) / fy  # image y down → camera Y up

    if y >= 0:
        return None

    Z = -CAMERA_HEIGHT / y
    X = x * Z

    return np.sqrt(X**2 + Z**2)

# ============================================================
# MAIN PIPELINE
# ============================================================
TP = FP = FN = 0
gt_all = []
est_all = []

log = open(os.path.join(LOG_OUT_PATH, "results.txt"), "w")

for img_name in sorted(os.listdir(IMAGE_PATH)):
    img_id = os.path.splitext(img_name)[0]

    img = cv2.imread(os.path.join(IMAGE_PATH, img_name))
    K = load_calibration(os.path.join(CALIB_PATH, img_id + ".txt"))
    gt_objects = load_labels(os.path.join(LABEL_PATH, img_id + ".txt"))

    # ---------------- DRAW GT BOXES (GREEN) ----------------
    for cls, gt_box, _ in gt_objects:
        if cls != "Car":
            continue
        x1, y1, x2, y2 = map(int, gt_box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ---------------- YOLO DETECTION ----------------
    results = model(img)[0]
    matched_gt = set()

    for det in results.boxes:
        cls_name = model.names[int(det.cls)]
        if cls_name != "car":
            continue

        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        pred_box = (x1, y1, x2, y2)

        best_iou = 0
        best_idx = -1

        for i, (_, gt_box, _) in enumerate(gt_objects):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= IOU_THRESHOLD:
            TP += 1
            matched_gt.add(best_idx)

            gt_dist = gt_objects[best_idx][2]
            est_dist = estimate_depth(K, pred_box)

            if est_dist is not None:
                gt_all.append(gt_dist)
                est_all.append(est_dist)
                label = f"{est_dist:.1f}m IoU={best_iou:.2f}"
                color = (0, 0, 255)  # RED → YOLO
            else:
                label = f"invalid IoU={best_iou:.2f}"
                color = (0, 0, 255)

            log.write(f"{img_id}, GT={gt_dist:.2f}, EST={est_dist}, IoU={best_iou:.2f}\n")

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, label, (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            FP += 1

    FN += len(gt_objects) - len(matched_gt)
    cv2.imwrite(os.path.join(IMG_OUT_PATH, img_id + ".png"), img)

log.close()

# ============================================================
# METRICS
# ============================================================
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("\n================ FINAL RESULTS ================")
print(f"TP={TP}  FP={FP}  FN={FN}")
print(f"Precision={precision:.3f}")
print(f"Recall={recall:.3f}")
print("================================================")

# ============================================================
# PLOTS
# ============================================================
gt_all = np.array(gt_all)
est_all = np.array(est_all)

plt.figure(figsize=(6, 6))
plt.scatter(gt_all, est_all)
plt.plot([0, max(gt_all)], [0, max(gt_all)], 'r--')
plt.xlabel("Ground Truth Distance (m)")
plt.ylabel("Estimated Distance (m)")
plt.title("GT vs Estimated Distance")
plt.grid(True)
plt.savefig(os.path.join(PLOT_OUT_PATH, "gt_vs_est.png"), dpi=300)
plt.close()

error = np.abs(gt_all - est_all)
plt.figure(figsize=(6, 4))
plt.scatter(gt_all, error)
plt.xlabel("Ground Truth Distance (m)")
plt.ylabel("Absolute Error (m)")
plt.title("Depth Error vs Distance")
plt.grid(True)
plt.savefig(os.path.join(PLOT_OUT_PATH, "error_vs_distance.png"), dpi=300)
plt.close()


## For each image in the KITTI selection, I run a pretrained YOLO 
# detector to detect cars, match detections with ground truth 
# using IoU thresholding, evaluate precision and recall, and 
# estimate car distance using camera intrinsics and a ground-plane 
# intersection model.