from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

prev_box = None
motion_threshold = 20  # Pixels — adjust sensitivity

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO prediction on frame (only class 0 = person)
    results = model.predict(frame, classes=[0], conf=0.3, verbose=False)[0]

    person_moving = False
    for box in results.boxes:
        # Get current person's box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # center of box

        if prev_box is not None:
            px, py = prev_box
            distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

            if distance > motion_threshold:
                person_moving = True
        prev_box = (cx, cy)

        # Draw box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Print movement status in terminal
    if results.boxes:
        if person_moving:
            print(" Person is moving")
        else:
            print(" Person is standing still")
    else:
        print(" No person detected")
        prev_box = None  # Reset if no detection

    # Show frame
    cv2.imshow("Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
