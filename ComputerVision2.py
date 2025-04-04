import numpy as np
import cv2
import os
import time

# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas a los archivos de configuración y pesos de YOLO
config_path = os.path.join(current_dir, 'yolov3.cfg')
weights_path = os.path.join(current_dir, 'yolov3.weights')
names_path = os.path.join(current_dir, 'coco.names')

# Verificar si los archivos existen
required_files = [config_path, weights_path, names_path]
for file_path in required_files:
    if not os.path.isfile(file_path):
        print(f"Error: No se encontró el archivo '{file_path}'")
        print("Por favor, descarga los archivos necesarios para YOLO:")
        print("- yolov3.cfg: archivo de configuración")
        print("- yolov3.weights: archivo de pesos (aproximadamente 240MB)")
        print("- coco.names: archivo con nombres de clases")
        exit()

# Cargar la red YOLO pre-entrenada y configurar para CPU (o GPU si se dispone de ella)
try:
    net = cv2.dnn.readNet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Red YOLO cargada exitosamente")
except cv2.error as e:
    print(f"Error al cargar la red YOLO: {e}")
    exit()

# Cargar las clases
try:
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"Se cargaron {len(classes)} clases")
except Exception as e:
    print(f"Error al cargar las clases: {e}")
    exit()

# Obtener los nombres de las capas de salida
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Iniciar la captura de video desde la cámara web
print("Iniciando cámara web...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara web")
    exit()

print("Detección de personas iniciada. Presiona 'q' para salir.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el cuadro de la cámara")
            break

        # Reducir tamaño para acelerar el procesamiento (usar resolución menor si es posible)
        frame_resized = cv2.resize(frame, (320, 320))
        height, width, _ = frame_resized.shape

        # Crear blob para la red neuronal
        blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        start_time = time.time()
        outs = net.forward(output_layers)
        end_time = time.time()

        class_ids = []
        confidences = []
        boxes = []

        # Procesar las detecciones
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # Solo detectar personas con confianza mayor al 50%
                if confidence > 0.5 and classes[class_id] == 'person':
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Aplicar Non-Maximum Suppression (NMS)
        try:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        except Exception:
            indexes = []

        # Dibujar los rectángulos y etiquetas
        if len(indexes) > 0:
            if isinstance(indexes, tuple):
                indexes = indexes[0]
            for i in indexes.flatten() if hasattr(indexes, 'flatten') else indexes:
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_resized, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        num_persons = len(indexes) if isinstance(indexes, (list, np.ndarray)) else 0
        fps_text = f"FPS: {1/(end_time - start_time):.2f}" if (end_time - start_time) > 0 else "FPS: N/A"
        cv2.putText(frame_resized, f"Personas detectadas: {num_persons}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame_resized, fps_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow('Detección de Personas en Tiempo Real', frame_resized)

        # Pequeña pausa para limitar el FPS y evitar saturar la CPU
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Saliendo del programa...")
            break
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Interrupción manual detectada. Saliendo...")

finally:
    cap.release()
    cv2.destroyAllWindows()