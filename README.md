# distance-estimation-of-car-from-camera

Depth Estimation (KITTI Selection)

# 1. Problem Statement
1. Detect "cars" in monocular images using a pre-trained object detector (YOLO).
2. Evaluate detection performance using **Intersection over Union (IoU)**, **precision**, and **recall**.
3. Estimate the **distance of detected cars** using **camera intrinsic calibration** and **ground-plane geometry**.
4. Compare the estimated distances with **ground truth distances** provided in the KITTI selection.


# 2. Input Data

The task uses the dataset folder **`KITTI_Selection`**, which contains:

 2.1 Images

* RGB images (`.png`)
* One image per scene
* Resolution varies (KITTI format)

 2.2 Camera Calibration Files

* One calibration file per image (`.txt`)
* Each file contains the **camera intrinsic matrix**:

<img width="233" height="116" alt="image" src="https://github.com/user-attachments/assets/78e2e99b-be3e-4567-b257-36d13ec58199" />


# 2.3 Ground Truth Labels

Each label file contains one line per object:

```
<Class> xmin ymin xmax ymax distance
```

Where:

* `Class` = object category (e.g. Car)
* `(xmin, ymin, xmax, ymax)` = 2D bounding box
* `distance` = ground truth distance (meters), precomputed from 3D KITTI annotations

---

# 3. Assumptions

The following assumptions are explicitly made (as required by the task):

1. The scene can be locally approximated as a **flat ground plane**.
2. The camera is:

   * mounted at a fixed height
   h = 1.65 m
   * aligned with the ground (no pitch or roll).
3. Distance is measured as **planar distance on the ground**, not Euclidean distance in 3D.
4. Only **cars** are considered for detection and evaluation.
5. Each ground-truth object can be matched to **at most one** detection (one-to-one matching).

---

## 4. Method Overview (Pipeline)

For each image, the following steps are executed **in order**:

1. Load image, calibration matrix, and ground-truth labels.
2. Run YOLO object detection.
3. Filter YOLO detections to keep only cars.
4. Match YOLO detections to ground truth using **maximum IoU**.
5. Classify detections into **TP / FP / FN** using an IoU threshold.
6. Estimate distance for **true positives only**.
7. Visualize results (GT boxes in green, YOLO boxes in red).
8. Compute precision, recall, and distance error plots.

---

## 5. Object Detection

### 5.1 YOLO Inference

A pre-trained YOLOv8 model is used:

```python
model = YOLO("yolov8n.pt")
results = model(img)[0]
```

Each detection provides:

* Bounding box ( b_p = (x_{\min}, y_{\min}, x_{\max}, y_{\max}) )
* Predicted class ID

Only detections with class **`car`** are considered.

---

## 6. Intersection over Union (IoU)

### 6.1 Definition

For a predicted bounding box (B_p) and a ground-truth box (B_g):

<img width="847" height="212" alt="image" src="https://github.com/user-attachments/assets/5247ca38-340c-4b53-90d1-8b8561666766" />


### 6.2 Intersection Computation

<img width="887" height="248" alt="image" src="https://github.com/user-attachments/assets/a340a790-eef1-4966-bd6c-6ed1ae8109ef" />

---

## 7. Matching Strategy and Evaluation

### 7.1 IoU Threshold

A detection is considered correct if:

<img width="976" height="72" alt="image" src="https://github.com/user-attachments/assets/ec956305-a0ab-417b-9c6f-923c2c4f94d0" />


### 7.2 Definitions

Let (P) be the set of predictions and (G) the set of ground-truth objects.

<img width="731" height="281" alt="image" src="https://github.com/user-attachments/assets/a6c72057-179a-40fb-85dd-345e1b3ce051" />


### 7.3 Precision and Recall

<img width="658" height="166" alt="image" src="https://github.com/user-attachments/assets/5ae06dbe-7f80-49e6-8152-f2b3d2606c95" />

---

## 8. Depth Estimation (Monocular Geometry)

Depth estimation is performed **only for true positives**.

### 8.1 Image Point Selection

For a detected bounding box:

* Bottom-center pixel is selected:
  <img width="1038" height="88" alt="image" src="https://github.com/user-attachments/assets/101b343f-bf52-4b7b-8b87-ebfe16e29a83" />


This point approximates the contact point of the object with the ground.

---

### 8.2 Inverse Projection (Pixel → Ray)

Using the intrinsic matrix:
<img width="562" height="102" alt="image" src="https://github.com/user-attachments/assets/f3799190-92d6-4f49-8eac-48f3e6b2cf59" />


This yields a 3D ray direction:

[
(X, Y, Z) = (xZ, yZ, Z)
]

---

### 8.3 Ground Plane Intersection

Ground plane equation:

Y = -h

Substituting into the ray equation:

<img width="1057" height="135" alt="image" src="https://github.com/user-attachments/assets/8507e40c-315a-4530-81fd-629cdcd91e38" />


### 8.4 Distance Computation

Planar distance on the ground:

<img width="1148" height="80" alt="image" src="https://github.com/user-attachments/assets/c0378e73-6e87-4209-b11b-a14338c0b992" />


This matches the definition used to compute the ground-truth distances.

---

## 9. Visualization

* **Ground truth bounding boxes**: green
* **YOLO detections**: red
* Distance and IoU values are annotated next to detections.

OpenCV functions used:

```python
cv2.rectangle()
cv2.putText()
```

---

## 10. Output

The program generates:

### 10.1 Annotated Images

Saved to:

```
Output/images/
```

Each image contains:

* Green GT boxes
* Red YOLO boxes
* Estimated distance and IoU

---

### 10.2 Logs

Saved to:

```
Output/logs/results.txt
```

Contains per-detection values:

```
image_id, GT distance, estimated distance, IoU
```

---

### 10.3 Plots

Saved to:

```
Output/plots/
```

1. **GT vs Estimated Distance**
2. **Absolute Error vs Distance**

These plots are used to analyze systematic errors and failure cases.

---

## 11. Discussion of Failure Cases

Large errors typically occur due to:

* Inaccurate bounding boxes (partial occlusions)
* Bottom of bounding box not lying on the ground
* Distant objects (small pixel error → large depth error)
* Assumption of flat ground violated

---

## 12. Conclusion

This task demonstrates how:

* Object detection can be quantitatively evaluated using IoU, precision, and recall.
* Monocular distance estimation is possible using camera intrinsics and geometric assumptions.
* Detection quality directly affects depth estimation accuracy.



