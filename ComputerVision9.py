import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np
import torch
import cv2
import threading
import os
from torchvision import models
import json


# Cargar modelos solo cuando sea necesario
class ModelManager:
    def __init__(self):
        self.yolo_model = None
        self.classification_model = None
        self.segmentation_model = None
        self.imagenet_labels = None

    def get_yolo(self):
        if self.yolo_model is None:
            print("Cargando modelo YOLO...")
            self.yolo_model = YOLO("yolov8n.pt")
        return self.yolo_model

    def get_classification_model(self):
        if self.classification_model is None:
            print("Cargando modelo de clasificación...")
            self.classification_model = models.resnet18(pretrained=True)
            self.classification_model.eval()

            # Cargar etiquetas de ImageNet si existe el archivo
            try:
                with open('imagenet_labels.json', 'r') as f:
                    self.imagenet_labels = json.load(f)
            except FileNotFoundError:
                self.imagenet_labels = ["Clase " + str(i) for i in range(1000)]
        return self.classification_model

    def get_segmentation_model(self):
        if self.segmentation_model is None:
            print("Cargando modelo de segmentación...")
            self.segmentation_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
            self.segmentation_model.eval()
        return self.segmentation_model

    def get_imagenet_label(self, index):
        if self.imagenet_labels:
            return self.imagenet_labels[index]
        return f"Clase {index}"


# Preprocesamiento de imágenes
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# Clase principal de la aplicación
class VisionApp:
    def __init__(self, root):
        self.root = root
        self.model_manager = ModelManager()
        self.setup_ui()

        # Variables para seguimiento
        self.tracking_object = False
        self.track_window = None
        self.roi_hist = None

        # Variable para controlar la ejecución de los procesos
        self.running = True
        self.active_frame = None

        # Iniciar captura de video
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return

        # Configurar hilos para procesamiento paralelo
        self.setup_threads()

    def setup_ui(self):
        # Configuración de la interfaz principal
        self.root.geometry("1200x700")
        self.root.title("Visión Computacional en Vivo")

        # Menú lateral
        self.menu = ctk.CTkFrame(self.root, width=200)
        self.menu.pack(side="left", fill="y")

        # Frames para cada funcionalidad
        self.frame_classification = ctk.CTkFrame(self.root)
        self.frame_detection = ctk.CTkFrame(self.root)
        self.frame_segmentation = ctk.CTkFrame(self.root)
        self.frame_tracking = ctk.CTkFrame(self.root)

        # Botones del menú
        self.btn_classify = ctk.CTkButton(self.menu, text="Clasificación",
                                          command=lambda: self.show_frame(self.frame_classification))
        self.btn_classify.pack(pady=10, padx=20, fill="x")

        self.btn_detect = ctk.CTkButton(self.menu, text="Detección",
                                        command=lambda: self.show_frame(self.frame_detection))
        self.btn_detect.pack(pady=10, padx=20, fill="x")

        self.btn_segment = ctk.CTkButton(self.menu, text="Segmentación",
                                         command=lambda: self.show_frame(self.frame_segmentation))
        self.btn_segment.pack(pady=10, padx=20, fill="x")

        self.btn_track = ctk.CTkButton(self.menu, text="Seguimiento",
                                       command=lambda: self.show_frame(self.frame_tracking))
        self.btn_track.pack(pady=10, padx=20, fill="x")

        # Configurar frames individuales
        self.setup_classification_frame()
        self.setup_detection_frame()
        self.setup_segmentation_frame()
        self.setup_tracking_frame()

        # Mostrar el frame de clasificación por defecto
        self.show_frame(self.frame_classification)

    def setup_classification_frame(self):
        # Video feed
        self.lbl_classification_video = ctk.CTkLabel(self.frame_classification, text="")
        self.lbl_classification_video.pack(pady=10)

        # Resultados
        self.result_text = ctk.CTkLabel(self.frame_classification,
                                        text="Esperando clasificación...",
                                        font=("Arial", 16))
        self.result_text.pack(pady=20)

        # Top 3 predicciones
        self.top_predictions = ctk.CTkLabel(self.frame_classification,
                                            text="",
                                            font=("Arial", 14))
        self.top_predictions.pack(pady=10)

    def setup_detection_frame(self):
        self.lbl_detected = ctk.CTkLabel(self.frame_detection, text="")
        self.lbl_detected.pack(pady=10)

        self.detection_info = ctk.CTkLabel(self.frame_detection,
                                           text="",
                                           font=("Arial", 14))
        self.detection_info.pack(pady=10)

    def setup_segmentation_frame(self):
        # Frame para mostrar imagen original y segmentada lado a lado
        self.segmentation_frame = ctk.CTkFrame(self.frame_segmentation)
        self.segmentation_frame.pack(pady=10, fill="both", expand=True)

        # Imagen original
        self.lbl_original = ctk.CTkLabel(self.segmentation_frame, text="Imagen Original")
        self.lbl_original.pack(pady=5, side="left", padx=10)

        # Imagen segmentada
        self.lbl_segmented = ctk.CTkLabel(self.segmentation_frame, text="Segmentación")
        self.lbl_segmented.pack(pady=5, side="right", padx=10)

    def setup_tracking_frame(self):
        self.lbl_tracked = ctk.CTkLabel(self.frame_tracking, text="")
        self.lbl_tracked.pack(pady=10)

        self.btn_select = ctk.CTkButton(self.frame_tracking,
                                        text="Seleccionar Objeto",
                                        command=self.select_object)
        self.btn_select.pack(pady=10)

        self.tracking_info = ctk.CTkLabel(self.frame_tracking,
                                          text="Ningún objeto seleccionado para seguimiento",
                                          font=("Arial", 14))
        self.tracking_info.pack(pady=10)

    def show_frame(self, frame):
        for f in [self.frame_classification, self.frame_detection,
                  self.frame_segmentation, self.frame_tracking]:
            f.pack_forget()
        frame.pack(fill="both", expand=True)
        self.active_frame = frame

    def setup_threads(self):
        # Iniciar threads para cada tipo de procesamiento
        self.classification_thread = threading.Thread(target=self.classification_loop)
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.segmentation_thread = threading.Thread(target=self.segmentation_loop)
        self.tracking_thread = threading.Thread(target=self.tracking_loop)

        # Configurar como daemon para que terminen cuando el programa principal termine
        self.classification_thread.daemon = True
        self.detection_thread.daemon = True
        self.segmentation_thread.daemon = True
        self.tracking_thread.daemon = True

        # Iniciar threads
        self.classification_thread.start()
        self.detection_thread.start()
        self.segmentation_thread.start()
        self.tracking_thread.start()

    def classification_loop(self):
        while self.running:
            if self.active_frame == self.frame_classification:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Procesar para clasificación
                try:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Mostrar el frame en la interfaz
                    display_img = img.resize((480, 360))
                    photo = ImageTk.PhotoImage(display_img)

                    # Actualizar UI de forma segura
                    self.root.after(0, lambda p=photo: self.lbl_classification_video.configure(image=p))
                    self.root.after(0, lambda p=photo: setattr(self.lbl_classification_video, 'image', p))

                    # Clasificar la imagen
                    model = self.model_manager.get_classification_model()
                    with torch.no_grad():
                        input_tensor = preprocess_image(img)
                        outputs = model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs[0], dim=0)

                        # Obtener top 3 predicciones
                        top3_prob, top3_indices = torch.topk(probs, 3)

                        # Clase con mayor probabilidad
                        top_class = top3_indices[0].item()
                        top_prob = top3_prob[0].item() * 100
                        class_name = self.model_manager.get_imagenet_label(top_class)

                        # Mostrar resultados
                        result_str = f"Clase detectada: {class_name} ({top_prob:.2f}%)"

                        # Preparar top 3 predicciones
                        top3_text = "Top 3 predicciones:\n"
                        for i in range(3):
                            idx = top3_indices[i].item()
                            prob = top3_prob[i].item() * 100
                            label = self.model_manager.get_imagenet_label(idx)
                            top3_text += f"{i + 1}. {label} ({prob:.2f}%)\n"

                        # Actualizar UI
                        self.root.after(0, lambda s=result_str: self.result_text.configure(text=s))
                        self.root.after(0, lambda s=top3_text: self.top_predictions.configure(text=s))
                except Exception as e:
                    print(f"Error en clasificación: {e}")

            # Pequeño delay para no saturar la CPU
            cv2.waitKey(30)

    def detection_loop(self):
        while self.running:
            if self.active_frame == self.frame_detection:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                try:
                    # Detección de objetos con YOLO
                    model = self.model_manager.get_yolo()
                    results = model(frame)

                    # Procesar resultados
                    annotated_frame = results[0].plot()

                    # Convertir para mostrar en UI
                    img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                    photo = ImageTk.PhotoImage(img)

                    # Actualizar UI
                    self.root.after(0, lambda p=photo: self.lbl_detected.configure(image=p))
                    self.root.after(0, lambda p=photo: setattr(self.lbl_detected, 'image', p))

                    # Mostrar información de detecciones
                    if results[0].boxes:
                        num_objects = len(results[0].boxes)
                        classes = results[0].boxes.cls
                        names = [results[0].names[int(c)] for c in classes]
                        class_counts = {}
                        for name in names:
                            if name in class_counts:
                                class_counts[name] += 1
                            else:
                                class_counts[name] = 1

                        info_text = f"Detectados {num_objects} objetos\n"
                        for cls, count in class_counts.items():
                            info_text += f"{cls}: {count}\n"

                        self.root.after(0, lambda s=info_text: self.detection_info.configure(text=s))
                    else:
                        self.root.after(0, lambda: self.detection_info.configure(text="No se detectaron objetos"))
                except Exception as e:
                    print(f"Error en detección: {e}")

            cv2.waitKey(30)

    def segmentation_loop(self):
        while self.running:
            if self.active_frame == self.frame_segmentation:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                try:
                    # Preparar imagen para segmentación
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    input_tensor = preprocess_image(img)

                    # Realizar segmentación
                    model = self.model_manager.get_segmentation_model()
                    with torch.no_grad():
                        output = model(input_tensor)['out'][0]
                        output_predictions = output.argmax(0).byte().cpu().numpy()

                    # Aplicar mapa de colores para mejor visualización
                    r = np.zeros_like(output_predictions).astype(np.uint8)
                    g = np.zeros_like(output_predictions).astype(np.uint8)
                    b = np.zeros_like(output_predictions).astype(np.uint8)

                    for i in range(21):  # 21 clases en PASCAL VOC
                        r[output_predictions == i] = np.random.randint(0, 255)
                        g[output_predictions == i] = np.random.randint(0, 255)
                        b[output_predictions == i] = np.random.randint(0, 255)

                    colored_mask = np.stack([r, g, b], axis=2)

                    # Redimensionar para mostrar
                    original_img = img.resize((400, 300))
                    segmented_img = Image.fromarray(colored_mask).resize((400, 300))

                    # Convertir para UI
                    photo_original = ImageTk.PhotoImage(original_img)
                    photo_segmented = ImageTk.PhotoImage(segmented_img)

                    # Actualizar UI
                    self.root.after(0, lambda p=photo_original: self.lbl_original.configure(image=p))
                    self.root.after(0, lambda p=photo_original: setattr(self.lbl_original, 'image', p))

                    self.root.after(0, lambda p=photo_segmented: self.lbl_segmented.configure(image=p))
                    self.root.after(0, lambda p=photo_segmented: setattr(self.lbl_segmented, 'image', p))
                except Exception as e:
                    print(f"Error en segmentación: {e}")

            cv2.waitKey(100)  # Más intervalo para segmentación (proceso pesado)

    def tracking_loop(self):
        while self.running:
            if self.active_frame == self.frame_tracking:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                try:
                    # Realizar seguimiento si hay un objeto seleccionado
                    if self.tracking_object and self.track_window is not None:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                        # Calcular retro-proyección
                        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

                        # Aplicar meanShift para ubicar el nuevo centroide
                        ret, self.track_window = cv2.meanShift(dst, self.track_window,
                                                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

                        # Dibujar rectángulo
                        x, y, w, h = self.track_window
                        tracking_frame = cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Mostrar información de seguimiento
                        info_text = f"Objeto seguido en posición: ({x}, {y})\nTamaño: {w}x{h}"
                        self.root.after(0, lambda s=info_text: self.tracking_info.configure(text=s))
                    else:
                        tracking_frame = frame.copy()
                        self.root.after(0, lambda: self.tracking_info.configure(
                            text="Haz clic en 'Seleccionar Objeto' para iniciar el seguimiento"))

                    # Mostrar frame en UI
                    img = Image.fromarray(cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2RGB))
                    photo = ImageTk.PhotoImage(img)
                    self.root.after(0, lambda p=photo: self.lbl_tracked.configure(image=p))
                    self.root.after(0, lambda p=photo: setattr(self.lbl_tracked, 'image', p))
                except Exception as e:
                    print(f"Error en seguimiento: {e}")

            cv2.waitKey(30)

    def select_object(self):
        # Pausar el seguimiento actual
        self.tracking_object = False

        # Capturar frame actual
        ret, frame = self.cap.read()
        if not ret:
            print("Error al capturar frame para selección de objeto")
            return

        # Crear ventana para selección de ROI
        r = cv2.selectROI("Selecciona un objeto y presiona ENTER", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Selecciona un objeto y presiona ENTER")

        if r[2] > 0 and r[3] > 0:  # Si se seleccionó una región válida
            self.track_window = r
            roi = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

            # Convertir ROI a HSV para mejor seguimiento
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Calcular histograma de la región seleccionada
            self.roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
            cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

            # Activar seguimiento
            self.tracking_object = True

            # Actualizar información
            self.tracking_info.configure(text="Objeto seleccionado para seguimiento")
        else:
            self.tracking_info.configure(text="No se seleccionó ningún objeto válido")

    def on_closing(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


# Iniciar aplicación
if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    app = ctk.CTk()
    vision_app = VisionApp(app)
    app.protocol("WM_DELETE_WINDOW", vision_app.on_closing)
    app.mainloop()