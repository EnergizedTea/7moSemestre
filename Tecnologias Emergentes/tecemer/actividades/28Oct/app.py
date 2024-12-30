from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("yolov8x.pt")

IMAGE_PATH = "bus.jpg"
img = cv2.imread(IMAGE_PATH)

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    # Write the frame to the output file
    out.write(frame)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

    results = model.predict(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = box.cls[0]
            class_name = model.names[int(class_id)]
            
            # Draw bounding box and label on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    
    # Display the frame with detection results
    cv2.imshow("YOLO Detection", frame)

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()


'''results = model.predict(image)

#print(results)
for result in results:
    boxes = result.boxes
    # print(boxes)
    for box in boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = box.cls[0]
        class_name = model.names[int(class_id)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0,), 4)
        cv2.putText(img, f"{class_name}: {confidence:.2F}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 4)
        
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()'''