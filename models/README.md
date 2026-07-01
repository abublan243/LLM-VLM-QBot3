# Model Weights

This directory stores ML model weights required by QBot3. These files are **not tracked by Git** due to their size.

## Required Models

| File | Size | Description | Source |
|------|------|-------------|--------|
| `yolo11l.pt` | ~50 MB | YOLOv11 Large — primary object detector | [Ultralytics](https://docs.ultralytics.com/models/yolo11/) |
| `yolov8s-world.pt` | ~26 MB | YOLO-World Small — open-vocabulary detector | [Ultralytics](https://docs.ultralytics.com/models/yolo-world/) |

## Optional / Custom Models

| File | Size | Description |
|------|------|-------------|
| `military_yolov8.pt` | ~6 MB | Fine-tuned YOLOv8 for military object classes |
| `military_rfdetr.pth` | ~346 MB | RT-DETR model for military object detection |

## Download Instructions

### YOLO models (via Ultralytics)

```bash
pip install ultralytics

# Download automatically on first use, or manually:
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
python -c "from ultralytics import YOLO; YOLO('yolov8s-world.pt')"
```

Place the downloaded `.pt` files in the project root directory.

### Custom models

Contact the project maintainers for access to the fine-tuned military detection models.
