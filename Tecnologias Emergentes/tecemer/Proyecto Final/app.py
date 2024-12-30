import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Carga del modelo H5
model = tf.keras.models.load_model("modelo.h5")

# Función de preprocesamiento de la imagen
def preprocesamiento(image):
    imagen_resize = cv2.resize(image, (416, 416)) # El modelo se maneja con imágenes de 416x416
    imagen_normalizada = imagen_resize / 255.0 # Normalización de los valores de los pixeles
    return np.expand_dims(imagen_normalizada, axis=0)

# Se abre la cámara
cam = cv2.VideoCapture(0)

# Se obtienen las dimensiones del frame
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Se crea el objeto para guardar el video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    if not ret:
        break

    # Escribir el frame en el video
    out.write(frame)

    # Preprocesamiento del frame para predicción
    input_data = preprocesamiento(frame)

    # Se realiza la predicción
    predictions = model.predict(input_data)

    # Se procesan las predicciones
    for prediction in predictions:
        x1, y1, x2, y2, confidence, class_id = map(int, prediction)
        class_name = f"Basura {class_id}"

        # Dibujar el rectángulo y el texto
        cv2.rectangle(frame, (x1, y1), (x2, y2), (112, 42, 10), 4)
        cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (214, 156, 21), 4)

    # Display the frame with detection results
    cv2.imshow("Custom H5 Model Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
