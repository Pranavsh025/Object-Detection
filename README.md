# Object_Detection

Vehicle detection using YOLOv8, trained on a custom vehicle dataset.

## What's in here

Loaded the dataset (uploaded as a zip and extracted), split it into train/valid/test, and loaded a pretrained YOLOv8n (nano) model as the starting point. Ran a baseline validation on it first, then fine-tuned it on the vehicle dataset for 30 epochs at 640x640 image size.

## Dataset

Used a small vehicle detection dataset (sourced from Roboflow) in YOLO format, covering 5 classes: bus, car, motorcycle, pedestrian, and semi. It came with only a train split, so it got randomly divided into train/valid/test (70/20/10) before training - around 360 images total.

## Files

- `Object_Detection.ipynb` - the notebook
- `dataset/` - the vehicle images and YOLO-format labels used for training

## Results

Trained for 30 epochs and the model ended up performing pretty well on the validation set:

- **mAP50**: 0.931
- **mAP50-95**: 0.744
- **Precision**: 0.93
- **Recall**: 0.858

Per-class breakdown showed some classes were much easier to detect than others - "bus" and "semi" hit near-perfect mAP50 (0.98+), while "pedestrian" lagged behind at 0.763 mAP50, likely because there were far fewer pedestrian instances in the dataset (14) compared to cars (128) or buses (28). Training the whole thing only took about 3 minutes thanks to using the small YOLOv8n model on a T4 GPU.

## Running it

```
pip install ultralytics
```

Written for Google Colab, so if running locally just replace the `files.upload()` step with a direct path to the `dataset` folder. The `data.yaml` file inside points to the train/valid/test folders and lists the class names.

## Notes

Used YOLOv8n (the smallest/fastest variant) here mainly for speed, and it still landed a solid 0.93 mAP50 - shows the small model is more than capable for a fairly small, clean dataset like this. The pedestrian class is the clear weak spot though, more training images for that class specifically would probably help more than just adding more epochs. Worth trying `yolov8s.pt` too if squeezing out a bit more accuracy matters more than inference speed.
