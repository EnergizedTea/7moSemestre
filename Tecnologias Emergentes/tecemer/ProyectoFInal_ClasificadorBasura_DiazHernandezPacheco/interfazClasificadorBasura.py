import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO
import threading
from datetime import datetime
import numpy as np


class TrashDetector:
    def __init__(self):
        # Configurar la ventana principal
        self.window = ctk.CTk()
        self.window.title("Detector de Residuos en Tiempo Real")
        self.window.geometry("1200x800")
        ctk.set_appearance_mode("dark")

        # Inicializar el modelo YOLO
        self.model = YOLO('yolo_waste_detection/yolov8_training95/weights/best.pt')
        self.is_detecting = False
        self.camera_active = False

        # Variables para FPS
        self.fps_start_time = None
        self.fps = 0
        self.frames_count = 0

        self.setup_ui()
        self.setup_camera()

    def setup_ui(self):
        # Frame principal
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Frame izquierdo para la cámara
        self.camera_frame = ctk.CTkFrame(self.main_frame)
        self.camera_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        # Etiqueta para mostrar el video
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="")
        self.camera_label.pack(padx=10, pady=10)

        # Frame derecho para controles y resultados
        self.control_frame = ctk.CTkFrame(self.main_frame, width=300)
        self.control_frame.pack(side="right", padx=10, pady=10, fill="both")

        # Título
        self.title_label = ctk.CTkLabel(
            self.control_frame,
            text="Control Panel",
            font=("Roboto", 24, "bold")
        )
        self.title_label.pack(pady=20)

        # Botones
        self.start_button = ctk.CTkButton(
            self.control_frame,
            text="Iniciar Detección",
            command=self.toggle_detection,
            font=("Roboto", 14),
            height=40
        )
        self.start_button.pack(pady=10, padx=20, fill="x")

        self.camera_button = ctk.CTkButton(
            self.control_frame,
            text="Cambiar Cámara",
            command=self.toggle_camera,
            font=("Roboto", 14),
            height=40
        )
        self.camera_button.pack(pady=10, padx=20, fill="x")

        # Frame para estadísticas
        self.stats_frame = ctk.CTkFrame(self.control_frame)
        self.stats_frame.pack(pady=20, padx=20, fill="x")

        # FPS Counter
        self.fps_label = ctk.CTkLabel(
            self.stats_frame,
            text="FPS: 0",
            font=("Roboto", 14)
        )
        self.fps_label.pack(pady=5)

        # Detecciones
        self.detections_label = ctk.CTkLabel(
            self.control_frame,
            text="Detecciones:",
            font=("Roboto", 16, "bold")
        )
        self.detections_label.pack(pady=10)

        # Lista de detecciones
        self.detections_text = ctk.CTkTextbox(
            self.control_frame,
            height=200,
            font=("Roboto", 12)
        )
        self.detections_text.pack(pady=10, padx=20, fill="x")

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def toggle_detection(self):
        if not self.is_detecting:
            self.is_detecting = True
            self.start_button.configure(text="Detener Detección")
            self.detection_thread = threading.Thread(target=self.detect_video)
            self.detection_thread.daemon = True
            self.detection_thread.start()
        else:
            self.is_detecting = False
            self.start_button.configure(text="Iniciar Detección")

    def toggle_camera(self):
        self.camera_active = not self.camera_active
        if self.camera_active:
            self.cap.release()
            self.cap = cv2.VideoCapture(1)
        else:
            self.cap.release()
            self.cap = cv2.VideoCapture(0)

    def update_fps(self):
        if self.fps_start_time is None:
            self.fps_start_time = datetime.now()
            self.frames_count = 0
            return

        self.frames_count += 1
        time_diff = (datetime.now() - self.fps_start_time).total_seconds()

        if time_diff >= 1.0:
            self.fps = self.frames_count / time_diff
            self.fps_label.configure(text=f"FPS: {self.fps:.1f}")
            self.fps_start_time = datetime.now()
            self.frames_count = 0

    def detect_video(self):
        reciclables = [
    "Aerosols", "Aluminum can", "Aluminum caps", "Cardboard", "Foil", 
    "Glass bottle", "Iron utensils", "Metal shavings", "Milk bottle",
    "Paper", "Paper bag", "Paper cups", "Paper shavings", "Plastic bottle", 
    "Plastic can", "Plastic canister", "Plastic caps", "Plastic cup", 
    "Plastic shaker", "Plastic shavings", "Plastic toys", "Postal packaging", 
    "Scrap metal", "Stretch film", "Tetra pack", "Tin", "Zip plastic bag"
    ]
        organicos = [
    "Cellulose", "Organic", "Wood"
    ]
        no_reciclables = [
    "Combined plastic", "Container for household chemicals", 
    "Disposable tableware", "Electronics", "Textile", 
    "Unknown plastic", "Ceramic", "Furniture", "Liquid"
    ]

        while self.is_detecting:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Realizar detección
            results = self.model(frame, conf=0.5)

            # Dibujar detecciones
            detection = results[0].plot()

            # Actualizar lista de detecciones
            detections = []
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = r.names[class_id]

                    if class_name in reciclables:
                        tipo_categoria = "Reciclable"
                    elif class_name in organicos:
                        tipo_categoria = "Organico"
                    elif class_name in no_reciclables:
                        tipo_categoria = "No Reciclable"
                    else:
                        tipo_categoria = "Unknown"

                    detections.append(f"{tipo_categoria}: {conf:.2f}")

            self.detections_text.delete("1.0", "end")
            self.detections_text.insert("1.0", "\n".join(detections))

            image = cv2.cvtColor(detection, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)

            self.camera_label.configure(image=photo)
            self.camera_label.image = photo

            self.update_fps()

    def run(self):
        self.window.mainloop()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    app = TrashDetector()
    app.run()
