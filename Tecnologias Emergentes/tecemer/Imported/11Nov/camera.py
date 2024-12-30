from ultralytics import YOLO
import cv2

# Load the trained model weights
model = YOLO("runs/detect/train/weights/best.pt")

# Use the model for real-time video detection
cam = cv2.VideoCapture(1)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up the video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(frame)

    # Process each detected object in the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = box.cls[0]
            class_name = model.names[int(class_id)]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

    # Write the processed frame to the output file
    out.write(frame)

    # Display the frame with detection results
    cv2.imshow("YOLO Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cam.release()
out.release()
cv2.destroyAllWindows()
