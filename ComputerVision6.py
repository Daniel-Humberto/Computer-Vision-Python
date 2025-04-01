from torchvision.models import ResNet50_Weights
from torchvision import transforms, models
from PIL import Image, ImageTk
from ultralytics import YOLO
import customtkinter as ctk
import numpy as np
import threading
import torch
import time
import cv2




# Dashboard de visión computacional con modelos de: clasificación, detección, segmentación y seguimiento de objetos
class VisionDashboard:


    # Constructor de la clase
    def __init__(self):

        # Configuración principal de la ventana
        self.root = ctk.CTk()
        self.root.title("Dashboard de Visión Computacional")
        self.root.geometry("1920x1080")

        # Cargar modelos de IA
        self.load_models()

        # Configurar elementos de la interfaz de usuario
        self.setup_ui()

        # Inicializar la captura de video desde la cámara
        self.cap = cv2.VideoCapture(0)

        # Variables de control de la aplicación
        self.running = True
        self.current_frame = None
        self.current_mode = "classification"
        self.tracker = None
        self.tracking_bbox = None

        # Iniciar hilos para captura y procesamiento
        self.start_processing_threads()


    # Carga de modelos de IA
    def load_models(self):
        
        # Modelo ResNet50 para clasificación de imágenes
        self.classification_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.classification_model.eval()

        # Modelo YOLOv8 para detección de objetos
        self.detection_model = YOLO('yolov8n.pt')

        # Modelo YOLOv8-seg para segmentación semántica
        self.segmentation_model = YOLO('yolov8n-seg.pt')

        # Preprocesamiento de imágenes para clasificación
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])


    # Configuración de la interfaz gráfica del dashboard
    def setup_ui(self):
        
        # Configurar layout de la cuadrícula
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Frame para el menú superior
        menu_frame = ctk.CTkFrame(self.root)
        menu_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # Botones para cambiar entre modos
        modes = [
            ("Clasificación", "classification"),
            ("Detección de Objetos", "detection"),
            ("Segmentación", "segmentation"),
            ("Seguimiento de Objetos", "tracking")
        ]

        # Crear botones para cada modo
        for i, (text, mode) in enumerate(modes):
            btn = ctk.CTkButton(
                menu_frame,
                text=text,
                command=lambda m=mode: self.change_mode(m)
            )
            btn.pack(side="left", padx=5, pady=5)

        # Frame principal para mostrar el video
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Canvas para renderizar el video
        self.main_canvas = ctk.CTkCanvas(self.main_frame)
        self.main_canvas.pack(expand=True, fill="both", padx=5, pady=5)

        # Etiqueta para mostrar resultados de procesamiento
        self.result_label = ctk.CTkLabel(self.main_frame, text="")
        self.result_label.pack(pady=5)


    # Cambia el modo de operación del dashboard
    def change_mode(self, mode):

        # Cambia el modo actual y limpia la etiqueta de resultados
        self.current_mode = mode
        self.result_label.configure(text="")

        # Inicializar tracker si se selecciona el modo de seguimiento
        if mode == "tracking":
            self.tracker = cv2.TrackerKCF_create()
            self.tracking_bbox = None


    # Inicia los hilos para captura de video y procesamiento
    def start_processing_threads(self):
        
        # Hilo para captura continua de video
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Hilo para procesamiento de frames
        self.processing_thread = threading.Thread(target=self.process_frame)
        self.processing_thread.daemon = True
        self.processing_thread.start()


    # Función para captura y procesamiento de video
    def update_video(self):
        
        # Configurar la cámara
        while self.running:

            # Leer frame de la cámara
            ret, frame = self.cap.read()
        
            # Verificar si se capturó un frame
            if ret:
                # Convertir de BGR a RGB para compatibilidad
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                time.sleep(0.03)  # Limitar a ~30 FPS


    # Función para procesar cada frame según el modo seleccionado
    def process_frame(self):
        
        # Procesar frames en un bucle
        while self.running:
            if self.current_frame is not None:

                # Copiar el frame actual para procesamiento
                frame = self.current_frame.copy()
                processed_frame = None

                # Clasificación de imágenes con ResNet50
                if self.current_mode == "classification":

                    # Clasificación de imágenes con ResNet50
                    image = Image.fromarray(frame)
                    input_tensor = self.transform(image)
                    input_batch = input_tensor.unsqueeze(0)

                    with torch.no_grad():
                        output = self.classification_model(input_batch)

                    _, predicted = torch.max(output, 1)
                    self.result_label.configure(text=f"Clase: {predicted.item()}")
                    processed_frame = frame
    
                # Detección de objetos con YOLOv8
                elif self.current_mode == "detection":
                    results = self.detection_model(frame)
                    processed_frame = results[0].plot()

                # Segmentación semántica con YOLOv8-seg
                elif self.current_mode == "segmentation":
                    results = self.segmentation_model(frame)
                    processed_frame = results[0].plot()

                # Seguimiento de objetos usando KCF Tracker
                elif self.current_mode == "tracking":

                    if self.tracking_bbox is None:

                        # Detectar objeto si no hay uno siendo seguido
                        results = self.detection_model(frame)

                        # Obtener primer objeto detectado para seguimiento
                        if len(results[0].boxes) > 0:
                            box = results[0].boxes[0].xyxy[0].numpy()
                            self.tracking_bbox = (
                                int(box[0]), int(box[1]),
                                int(box[2] - box[0]), int(box[3] - box[1])
                            )
                            self.tracker = cv2.TrackerKCF_create()
                            self.tracker.init(frame, self.tracking_bbox)
                    else:

                        # Actualizar posición del objeto seguido
                        success, bbox = self.tracker.update(frame)
                        
                        if success:
                            x, y, w, h = [int(v) for v in bbox]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        else:
                            self.tracking_bbox = None

                    processed_frame = frame

                # Actualizar display si hay un frame procesado
                if processed_frame is not None:
                    self.update_main_display(Image.fromarray(processed_frame))

            time.sleep(0.1)  # Controlar frecuencia de procesamiento


    # Actualiza el dashboard con la imagen o canvas procesada
    def update_main_display(self, image):
        
        # Obtener dimensiones del canvas
        canvas_width = self.main_canvas.winfo_width()
        canvas_height = self.main_canvas.winfo_height()

        # Calcular tamaño preservando relación de aspecto
        img_width, img_height = image.size
        aspect_ratio = img_width / img_height

        # Ajustar tamaño de imagen para que se ajuste al canvas en proporción alta o ancha del mismo
        if canvas_width / canvas_height > aspect_ratio:
            new_height = canvas_height
            new_width = int(canvas_height * aspect_ratio)
        else:
            new_width = canvas_width
            new_height = int(canvas_width / aspect_ratio)

        # Redimensionar imagen para ajustar al canvas
        resized_image = image.resize((new_width, new_height))
        photo = ImageTk.PhotoImage(resized_image)

        # Centrar imagen en canvas
        x_center = (canvas_width - new_width) // 2
        y_center = (canvas_height - new_height) // 2

        # Limpiar canvas y mostrar imagen
        self.main_canvas.delete("all")
        self.main_canvas.create_image(x_center, y_center, anchor="nw", image=photo)
        self.main_canvas.image = photo


    # Inicia el bucle principal de la aplicación
    def run(self):
        self.root.mainloop()


    # Limpia recursos al cerrar la aplicación
    def cleanup(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()


# Ejecutar la aplicación
if __name__ == "__main__":
    app = VisionDashboard()
    try:
        app.run()
    finally:
        app.cleanup()