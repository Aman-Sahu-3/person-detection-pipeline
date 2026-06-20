# Person Detection Pipeline

Detects people in real-time using YOLOv8n and OpenCV.
Triggers an alert when a person enters a defined restricted zone (ROI).

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows


pip install -r requirements.txt
```

## Run

```bash
python detect.py                       # webcam
python detect.py --source sample.mp4   # video file
```

Press `q` to quit.

---

## What You See on Screen

| Visual | Meaning |
|---|---|
| Yellow rectangle | ROI — the restricted zone |
| Green box | Person detected, outside ROI |
| Red box | Person detected, inside ROI |
| Red text at top | Alert active this frame |

## Approach

- YOLOv8n pretrained on COCO (80 classes; class 0 = person)
- ROI check uses bounding box center point
- Alert cooldown of 2 seconds prevents terminal spam

## Limitations

- ROI is hardcoded, no interactive drawing
- CPU only may slow down on high-resolution video
- Single video source at a time
