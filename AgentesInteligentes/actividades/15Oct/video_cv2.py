import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error: Failure to Access Camera.')

while True:
    ret, frame = cap.read()

    if not ret: 
        print("Error Unable to receive frame.")
        # Ret es un valor booleano de si fue o no capaz de capturar ese frame
        break

    cv2.imshow('Fabulous Title', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()