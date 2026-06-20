

import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO


CONF_THRESHOLD = 0.5
ALERT_COOLDOWN = 2.0 # Without this, "ALERT" would print 30 times per second (once per frame).
ROI = (190, 40, 550, 850) #   Default values work for a standard 640x480 webcam frame (rx1, ry1, rx2, ry2).
# ──────────────────────────────────────────────────────────────────────────────


# Load YOLOv8n model.

model = YOLO("yolov8n.pt")


def is_in_roi(box, roi):
    """
    Check if the CENTER POINT of a bounding box is inside the ROI rectangle.

    Use center point because:
    - Full overlap triggers if only a hand/foot enters the zone
    - Center point means the person's body midpoint must be inside

    Returns:
        bool: True if person center is inside ROI, False otherwise
    """
    bx1, by1, bx2, by2 = box      # bounding box corners
    rx1, ry1, rx2, ry2 = roi      # ROI corners

    # Calculate the center of the bounding box
    cx = (bx1 + bx2) // 2         # horizontal center
    cy = (by1 + by2) // 2         # vertical center

    # Check if center point is inside the ROI rectangle
    horizontal_inside = rx1 < cx < rx2
    vertical_inside   = ry1 < cy < ry2

    return horizontal_inside and vertical_inside


def draw_roi(frame, roi):
    """
    Draw the ROI zone on the frame Solid yellow border around the zone

    draw on a copy, then blend copy + original using addWeighted().

    """
    rx1, ry1, rx2, ry2 = roi

    # Step 1: Copy the current frame
    overlay = frame.copy()

    # Step 2: Draw a SOLID filled rectangle on the copy
    #   Color (0, 255, 255) = yellow
    #   -1 as thickness means "filled"
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 255), -1)

    # Step 3: Blend the overlay (15%) with original frame (85%)
    #   Result: the filled rectangle looks 15% visible = semi-transparent
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Step 4: Draw the solid border on the actual frame
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

    # Step 5: Add "ROI" text label at top-left corner of the zone
    cv2.putText(
        frame,
        "ROI",
        (rx1 + 5, ry1 + 20),           # position: slightly inside top-left
        cv2.FONT_HERSHEY_SIMPLEX,       # font style
        0.6,                            # font size
        (0, 255, 255),                  # color: yellow
        2                               # thickness
    )


def process_frame(frame, last_alert_time):
    """
    The core function — runs every single frame.

    What it does:
        1. Sends the frame to YOLOv8 for person detection
        2. For each detected person:
            Gets the bounding box coordinates
            Checks if person is inside ROI
            Draws colored box (red = in ROI, green = outside)
            Draws confidence label
        3. If any person is in ROI:
            Prints alert to terminal (with cooldown)
            Shows alert text on screen
    """
    alert_triggered = False

    # classes=[0]          → only detect class index 0, which is "person" in the COCO dataset YOLO was trained on
    # verbose=False        → stop YOLO from printing stats every frame

    results = model(
        frame,
        conf=CONF_THRESHOLD,
        classes=[0],
        verbose=False
    )[0]

    # Loop Over Every Detected Person 
    
    # results.boxes is a list of detected objects.
    # Each box has:
    #   .xyxy[0]  → tensor [x1, y1, x2, y2] in pixel coordinates
    #   .conf[0]  → confidence score between 0 and 1
    #   .cls[0]   → class index (always 0 here since we filtered for person class) 

    for box in results.boxes:

        # Convert tensor coordinates to plain Python integers
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Convert confidence tensor to plain Python float
        conf = float(box.conf[0])

        # Check if this person's center is inside the ROI
        in_roi = is_in_roi((x1, y1, x2, y2), ROI)

        # Choose color based on ROI status
        # OpenCV color format is BGR (Blue, Green, Red)
        #   (0, 0, 255)   = Red   → person is inside ROI (danger)
        #   (0, 255, 0)   = Green → person is outside ROI (safe)
        color = (0, 0, 255) if in_roi else (0, 255, 0)

        # bounding box rectangle around the person with thickness=2 wide border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw confidence label just above the top-left of the box
        cv2.putText(
            frame,
            f"Person {conf:.2f}",
            (x1, y1 - 8),              # 8 pixels above the box top edge
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,                       # font scale
            color,
            2                           # thickness
        )

        # Flag that at least one person is inside ROI this frame
        if in_roi:
            alert_triggered = True

    # ── Alert Logic ───────────────────────────────────────────────────────────
    current_time = time.time()

    if alert_triggered:

        # Terminal print: only if cooldown period has passed

        if current_time - last_alert_time >= ALERT_COOLDOWN:
            print("ALERT: Person in restricted area.")
            last_alert_time = current_time    # reset the timer

        cv2.putText(
            frame,
            "!! ALERT: Person in restricted area !!",
            (20, 40),                   # top-left corner of the screen
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),                # red
            2
        )

    return frame, last_alert_time


def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Person Detection Pipeline with ROI Alert"
    )
    parser.add_argument(
        "--source",
        default="0",
        help='Video source. Use 0 for webcam or a file path like video.mp4'
    )
    return parser.parse_args()


def main():
    """
    Entry point. Sets up the video source and runs the frame loop.
    """
    args = parse_args()

    # If source is "0", "1", "2" → convert to integer (webcam index)
    # If source is "video.mp4"   → keep as string (file path)
    source = int(args.source) if args.source.isdigit() else args.source

    # Open the video source
    cap = cv2.VideoCapture(source)

    # Check if it opened successfully
    if not cap.isOpened():
        raise RuntimeError(
            f"\nERROR: Could not open video source: '{source}'\n"
            "- For webcam: make sure it's connected and not used by another app\n"
            "- For video file: check the file path is correct\n"
        )

    # Read and print video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nSource     : {source}")
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.1f}")
    print(f"ROI zone   : {ROI}")
    print("\nRunning... Press 'q' inside the video window to quit.\n")

    last_alert_time = 0.0    # no alert has been printed yet


    # This loop runs continuously, one iteration per frame.

    while True:

        # Read the next frame from the video source
        # ret   = True if frame was read successfully, False if stream ended
        # frame = numpy array of shape (height, width, 3) containing pixel values
        ret, frame = cap.read()

        # If reading failed (stream ended or camera disconnected), exit
        if not ret:
            print("Stream ended or failed to read frame. Exiting.")
            break

        # Step 1: Draw the ROI zone on the frame 
        draw_roi(frame, ROI)

        # Step 2: Run detection, ROI check, draw boxes and alerts
        frame, last_alert_time = process_frame(frame, last_alert_time)

        # Step 3: Show the annotated frame in a window
        cv2.imshow("Person Detection Pipeline", frame)

        # Step 4: Wait 1ms then check if 'q' was pressed
        #   waitKey(1) is what makes the window actually render (crucial)
        #   Without it the window would freeze
        #   0xFF mask handles different keyboard/OS encodings
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received.")
            break


    cap.release()            # release webcam or close video file
    cv2.destroyAllWindows()  # close the display window


if __name__ == "__main__":
    main()
