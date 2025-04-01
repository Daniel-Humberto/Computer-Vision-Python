from torchvision import transforms, models
from PIL import Image, ImageTk
from ultralytics import YOLO
import customtkinter as ctk
import numpy as np
import threading
import torch
import time
import cv2




#  Dashboard de visión computacional con modelos de: clasificación, detección, segmentación y seguimiento de objetos
class VisionDashboard:


    # Constructor de la clase
    def __init__(self):

        # Configuración principal de la ventana
        self.root = ctk.CTk()
        self.root.title("Dashboard de Visión Computacional")
        self.root.geometry("1920x1080")

        # Cargar modelos de IA preentrenados
        self.load_models()

        # Configurar elementos de la interfaz gráfica
        self.setup_ui()

        # Inicializar cámara para captura de video
        self.cap = cv2.VideoCapture(0)

        # Variables de control para ejecución
        self.running = True
        self.current_frame = None

        # Iniciar hilos para procesamiento paralelo
        self.start_processing_threads()


    # Carga de modelos de IA
    def load_models(self):

        # Cargar modelo ResNet50 preentrenado para clasificación
        self.classification_model = models.resnet50(pretrained=True)
        self.classification_model.eval()

        # Cargar modelo YOLOv8 para detección de objetos
        self.detection_model = YOLO('yolov8n.pt')

        # Cargar modelo YOLOv8 para segmentación de imágenes
        self.segmentation_model = YOLO('yolov8n-seg.pt')

        # Definir transformaciones para preparar imágenes para clasificación
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])


    # Configuración de la interfaz gráfica del dashboard
    def setup_ui(self):

        # Configurar grid para layout de 2x2
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure((0, 1), weight=1)

        # Crear panel para clasificación (arriba-izquierda)
        self.classification_frame = ctk.CTkFrame(self.root)
        self.classification_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.classification_label = ctk.CTkLabel(self.classification_frame, text="Clasificación en Vivo")
        self.classification_label.pack(pady=5)
        self.classification_canvas = ctk.CTkCanvas(self.classification_frame, width=400, height=300)
        self.classification_canvas.pack(pady=5)
        self.classification_result = ctk.CTkLabel(self.classification_frame, text="")
        self.classification_result.pack(pady=5)

        # Crear panel para detección de objetos (arriba-derecha)
        self.detection_frame = ctk.CTkFrame(self.root)
        self.detection_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.detection_label = ctk.CTkLabel(self.detection_frame, text="Detección de Objetos")
        self.detection_label.pack(pady=5)
        self.detection_canvas = ctk.CTkCanvas(self.detection_frame, width=400, height=300)
        self.detection_canvas.pack(pady=5)

        # Crear panel para segmentación (abajo-izquierda)
        self.segmentation_frame = ctk.CTkFrame(self.root)
        self.segmentation_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.segmentation_label = ctk.CTkLabel(self.segmentation_frame, text="Segmentación")
        self.segmentation_label.pack(pady=5)
        self.segmentation_canvas = ctk.CTkCanvas(self.segmentation_frame, width=400, height=300)
        self.segmentation_canvas.pack(pady=5)

        # Crear panel para seguimiento de objetos (abajo-derecha)
        self.tracking_frame = ctk.CTkFrame(self.root)
        self.tracking_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.tracking_label = ctk.CTkLabel(self.tracking_frame, text="Seguimiento de Objetos")
        self.tracking_label.pack(pady=5)
        self.tracking_canvas = ctk.CTkCanvas(self.tracking_frame, width=400, height=300)
        self.tracking_canvas.pack(pady=5)


    # Función para iniciar hilos de procesamiento
    def start_processing_threads(self):

        # Iniciar hilo para captura continua de video
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Iniciar hilo para clasificación de imágenes
        self.classification_thread = threading.Thread(target=self.process_classification)
        self.classification_thread.daemon = True
        self.classification_thread.start()

        # Iniciar hilo para detección de objetos
        self.detection_thread = threading.Thread(target=self.process_detection)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        # Iniciar hilo para segmentación de imágenes
        self.segmentation_thread = threading.Thread(target=self.process_segmentation)
        self.segmentation_thread.daemon = True
        self.segmentation_thread.start()

        # Iniciar hilo para seguimiento de objetos
        self.tracking_thread = threading.Thread(target=self.process_tracking)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()


    # Función para captura y procesamiento de video
    def update_video(self):
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                time.sleep(0.03)  # Aproximadamente 30 FPS


    # Función para procesamiento de clasificación de imágenes
    def process_classification(self):
        
        while self.running:
            if self.current_frame is not None:

                # Convertir frame a formato PIL para procesamiento
                image = Image.fromarray(self.current_frame)

                # Preprocesar imagen para el modelo
                input_tensor = self.transform(image)
                input_batch = input_tensor.unsqueeze(0)

                # Realizar inferencia sin calcular gradientes
                with torch.no_grad():
                    output = self.classification_model(input_batch)

                # Obtener clase predicha (índice de mayor valor)
                _, predicted = torch.max(output, 1)

                # Actualizar la interfaz con el resultado
                self.update_classification_ui(image, f"Clase: {predicted.item()}")

            time.sleep(0.1)


    # Función para detección de objetos
    def process_detection(self):
        
        while self.running:
            if self.current_frame is not None:

                # Realizar detección con YOLOv8
                results = self.detection_model(self.current_frame)

                # Generar imagen con anotaciones de detecciones
                annotated_frame = results[0].plot()

                # Actualizar la interfaz con el resultado
                self.update_detection_ui(Image.fromarray(annotated_frame))

            time.sleep(0.1)


    # Función para segmentación de objetos
    def process_segmentation(self):
        
        while self.running:
            if self.current_frame is not None:

                # Realizar segmentación con YOLOv8-seg
                results = self.segmentation_model(self.current_frame)

                # Generar imagen con máscaras de segmentación
                annotated_frame = results[0].plot()

                # Actualizar la interfaz con el resultado
                self.update_segmentation_ui(Image.fromarray(annotated_frame))
                
            time.sleep(0.1)


    # Función para seguimiento de objetos
    def process_tracking(self):
        
        # Inicializar tracker CSRT de OpenCV
        tracker = cv2.TrackerCSRT_create()
        bbox = None

        while self.running:
            if self.current_frame is not None:
                frame = self.current_frame.copy()

                if bbox is None:
                    # Si no hay objeto siendo seguido, detectar nuevo objeto
                    results = self.detection_model(frame)
                    if len(results[0].boxes) > 0:
                        # Tomar primera detección para iniciar seguimiento
                        box = results[0].boxes[0].xyxy[0].numpy()
                        # Convertir formato (x1,y1,x2,y2) a (x,y,width,height)
                        bbox = (int(box[0]), int(box[1]),
                                int(box[2] - box[0]), int(box[3] - box[1]))
                        tracker.init(frame, bbox)
                else:
                    # Actualizar posición del objeto seguido
                    success, bbox = tracker.update(frame)
                    if success:
                        # Dibujar rectángulo alrededor del objeto
                        x, y, w, h = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Actualizar la interfaz con el resultado
                self.update_tracking_ui(Image.fromarray(frame))
            time.sleep(0.1)  # Controlar frecuencia de procesamiento


    # Función para actualizar la interfaz gráfica de clasificación
    def update_classification_ui(self, image, text):
        photo = ImageTk.PhotoImage(image.resize((400, 300)))
        self.classification_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.classification_canvas.image = photo
        self.classification_result.configure(text=text)


    # Función para actualizar la interfaz gráfica de detección
    def update_detection_ui(self, image):
        photo = ImageTk.PhotoImage(image.resize((400, 300)))
        self.detection_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.detection_canvas.image = photo


    # Función para actualizar la interfaz gráfica de segmentación
    def update_segmentation_ui(self, image):
        photo = ImageTk.PhotoImage(image.resize((400, 300)))
        self.segmentation_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.segmentation_canvas.image = photo


    # Función para actualizar la interfaz gráfica de seguimiento
    def update_tracking_ui(self, image):
        photo = ImageTk.PhotoImage(image.resize((400, 300)))
        self.tracking_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.tracking_canvas.image = photo


    # Función principal para la interfaz gráfica
    def run(self):
        self.root.mainloop()


    # Función para limpia recursos al cerrar la aplicación
    def cleanup(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()


# Punto de entrada principal
if __name__ == "__main__":
    app = VisionDashboard()
    try:
        app.run()
    finally:
        app.cleanup()