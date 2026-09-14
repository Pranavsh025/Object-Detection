# Blood_Cell_Detection

Blood cell detection using YOLOv8, fine-tuned on the BCCD dataset — achieving **0.87 mAP50** on the held-out test set.

## What's in here

Loaded the BCCD dataset (uploaded as a zip and extracted), inspected the folder structure and YOLO-format labels, then loaded a pretrained YOLOv8n model as the starting point. Ran a baseline evaluation on it first (it scores near-zero, since RBC/WBC/Platelets aren't COCO classes), then fine-tuned it on the blood cell dataset for 30 epochs at 640×640 with the backbone frozen (`freeze=10`) — a good setup for a small dataset like this one.

## Dataset

[BCCD (Blood Cell Count and Detection)](https://public.roboflow.com/object-detection/bccd) — Roboflow Public Datasets, MIT license.

| Split | Images |
|---|---|
| Train | 765 |
| Valid | 73 |
| Test | 36 |

**Classes (3):** `RBC` (Red Blood Cell), `WBC` (White Blood Cell), `Platelets`

## Model & training

- **Base model:** YOLOv8n (nano), COCO-pretrained
- **Epochs:** 30
- **Image size:** 640×640
- **Backbone:** frozen first 10 layers (`freeze=10`) — only the detection head is retrained
- **Framework:** [Ultralytics](https://github.com/ultralytics/ultralytics)

## Results (test set)

| Metric | Pretrained (baseline) | Fine-tuned |
|---|---|---|
| mAP50 | 0.00 | **0.87** |
| mAP50-95 | 0.00 | **0.60** |
| Precision | 0.00 | **0.79** |
| Recall | 0.01 | **0.88** |

**Per-class mAP50 (fine-tuned):**

| Class | mAP50 |
|---|---|
| WBC | 0.97 |
| RBC | 0.87 |
| Platelets | 0.78 |

The pretrained COCO model performs at essentially zero — expected, since none of its 80 classes are blood cells. Fine-tuning does all the work here. WBCs (large, distinct nucleus) are the easiest to detect; Platelets (small, low-contrast) are the hardest, which is typical for this dataset.

## Files

- `BCCD_Blood_Cell_Detection.ipynb` — full notebook: data loading, baseline eval, fine-tuning, evaluation, comparison, and sample prediction visualization
- `best.pt` — fine-tuned model weights
- `README.md` — this file

## How to run

1. Open the notebook in Google Colab (or locally with a GPU).
2. Download the dataset in YOLOv8 format from [Roboflow](https://public.roboflow.com/object-detection/bccd/3/download/yolov8) and upload the zip when prompted.
3. Run all cells top to bottom — it installs `ultralytics`, evaluates the baseline, fine-tunes for 30 epochs, evaluates again, and saves the best weights automatically.

To run inference with the already-trained weights instead of retraining:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("path/to/image.jpg", conf=0.25)
results[0].show()
```

## Credits

- Dataset: [BCCD Dataset](https://public.roboflow.com/object-detection/bccd) by Roboflow, MIT license
- Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
