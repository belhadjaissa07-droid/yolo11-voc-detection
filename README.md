
# Object Detection on Pascal VOC with YOLO11n

## 1. Problem
Detect and localize objects from 20 everyday classes (person, car, dog, chair, ...) in images.
Output: bounding boxes, class labels and confidence scores.
Main metric: mAP50-95.

## 2. Dataset
Pascal VOC (2007 + 2012), loaded and converted to YOLO format by Ultralytics.
- Train: VOC2007 + VOC2012 trainval
- Val and test: VOC2007 test (the same folder)

## 3. Method
- Model: YOLO11n, pretrained on COCO (transfer learning), head adapted to 20 classes
- Image size: 640, batch 16, seed 42
- Epochs: 5 (limited by free Colab compute)
- Default Ultralytics augmentation (mosaic, flip, HSV)
- Hardware: Google Colab T4 GPU

## 4. Results
| Model | imgsz | Epochs | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------|--------|-------|----------|-----------|--------|
| YOLO11n | 640 | 5 | TODO | TODO | TODO | TODO |

Per-class mAP50-95:
- Weakest: pottedplant 0.34, chair 0.45, boat 0.48, bottle 0.51, bird 0.55
- Strongest: bus 0.76, car 0.73, cat 0.72, horse 0.70, train 0.67

## 5. Error analysis
- Weak classes are mostly small, cluttered or variable-looking objects (hypothesis, not tested).
- The confusion matrix shows mix-ups between visually similar classes:
  cat/dog, cow/horse, motorbike/car, sofa/chair.
- Many false detections are labeled `person` (background column).

## 6. Limitations
- Only 5 epochs, so the model is undertrained and part of the weakness may come from that.
- Val and test are the same set, and `best.pt` was selected on it, so scores are slightly optimistic.
- Experiments not run: larger image size (800), bigger model (yolo11s), longer training.
- Only the 20 VOC classes are recognized.

## 7. Demo
Gradio app (screenshot: TODO). Upload an image and adjust the confidence threshold.

## 8. How to run
```bash
pip install ultralytics gradio
```
Load the weights with `YOLO("best.pt")` and call `model("image.jpg")`.

## 9. Future work
Train 50+ epochs, test `imgsz=800` and `yolo11s`, hold out a separate test set.
