import cv2
import os

types = ('M146', 'CANES', 'PENTEL')
classes = ('foupen', 'balpen', 'mecpen')
for f in range(len(classes)):
    for i in range(7):
        VIDEO_PATH = f'VIDEO/{types[f]}-{i+1}.MOV'
        OUTPUT_PATH = f'dataset/{classes[f]}'

        # Crear el directorio de salida si no existe
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        # Abrir el video
        cap = cv2.VideoCapture(VIDEO_PATH)

        # Verificar si el video se abrió correctamente
        if not cap.isOpened():
            print(f"Error: No pude abrir el video {VIDEO_PATH}.")
            exit()

        # Obtener el frame rate
        frame_rate = round(cap.get(cv2.CAP_PROP_FPS))

        # Verificar si el frame_rate es válido
        if frame_rate == 0:
            print("Error: El frame rate del video es 0. El archivo de video podría estar corrupto o mal codificado.")
            cap.release()
            exit()

        # Definir el intervalo de frames
        frame_interval = max(int(frame_rate) // 2, 1)

        print(f'Frame rate original: {frame_rate}')
        print(f'Intervalo de frames: {frame_interval}')

        i = 0
        saved_frames = 0

        print(f'Guardando frames en {OUTPUT_PATH}...')
        # Leer los frames del video
        while True:
            ret, frame = cap.read()

            if not ret:
                # Fin del video o error
                print("Fin del video o error al recibir el frame.")
                break

            # Guardar cada frame según el intervalo definido
            if i % frame_interval == 0:
                frame_path = f'{OUTPUT_PATH}/frame_{i}.jpg'
                cv2.imwrite(frame_path, frame)
                # print(f'Frame guardado: {frame_path}')
                saved_frames += 1

            i += 1

        print(f'{saved_frames} frames guardados en {OUTPUT_PATH}')
        cap.release()
        cv2.destroyAllWindows()
