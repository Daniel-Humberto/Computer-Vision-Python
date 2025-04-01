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

        # Inicialización principal de la aplicación
        self.root = ctk.CTk()
        self.root.title("Dashboard de Visión Computacional")
        self.root.geometry("1920x1080")

        # Carga de los modelos de IA
        self.load_models()

        # Configuración de la interfaz gráfica
        self.setup_ui()

        # Inicialización de la cámara web
        self.cap = cv2.VideoCapture(0)

        # Variables de control para la aplicación
        self.running = True
        self.current_frame = None
        self.current_process = 'Clasificación'

        # Inicio de hilos para procesamiento paralelo
        self.start_processing_threads()


    # Carga de modelos de IA
    def load_models(self):

        # Modelo de clasificación basado en ResNet50
        self.classification_model = models.resnet50(pretrained=True)
        self.classification_model.eval()

        # Modelo de detección de objetos YOLO
        self.detection_model = YOLO('yolov8n.pt')

        # Modelo de segmentación semántica YOLO
        self.segmentation_model = YOLO('yolov8n-seg.pt')

        # Preprocesamiento para imágenes de entrada al modelo de clasificación
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])


    # Configuración de la interfaz gráfica del dashboard
    def setup_ui(self):

        # Menú superior de navegación
        self.menu_frame = ctk.CTkFrame(self.root)
        self.menu_frame.pack(side="top", fill="x")

        # Botones para cambiar entre las diferentes funcionalidades
        self.classification_button = ctk.CTkButton(self.menu_frame, text="Clasificación", command=self.show_classification)
        self.classification_button.pack(side="left", padx=5)

        self.detection_button = ctk.CTkButton(self.menu_frame, text="Detección de Objetos", command=self.show_detection)
        self.detection_button.pack(side="left", padx=5)

        self.segmentation_button = ctk.CTkButton(self.menu_frame, text="Segmentación", command=self.show_segmentation)
        self.segmentation_button.pack(side="left", padx=5)

        self.tracking_button = ctk.CTkButton(self.menu_frame, text="Seguimiento de Objetos", command=self.show_tracking)
        self.tracking_button.pack(side="left", padx=5)

        # Panel de clasificación de imágenes
        self.classification_frame = ctk.CTkFrame(self.root)
        self.classification_frame.pack(fill="both", expand=True)
        self.classification_label = ctk.CTkLabel(self.classification_frame, text="Clasificación en Vivo")
        self.classification_label.pack(pady=5)
        self.classification_canvas = ctk.CTkCanvas(self.classification_frame, width=1000, height=500)
        self.classification_canvas.pack(pady=5)
        self.classification_result = ctk.CTkLabel(self.classification_frame, text="")
        self.classification_result.pack(pady=5)

        # Panel de detección de objetos
        self.detection_frame = ctk.CTkFrame(self.root)
        self.detection_label = ctk.CTkLabel(self.detection_frame, text="Detección de Objetos")
        self.detection_label.pack(pady=5)
        self.detection_canvas = ctk.CTkCanvas(self.detection_frame, width=800, height=600)
        self.detection_canvas.pack(pady=5)

        # Panel de segmentación semántica
        self.segmentation_frame = ctk.CTkFrame(self.root)
        self.segmentation_label = ctk.CTkLabel(self.segmentation_frame, text="Segmentación")
        self.segmentation_label.pack(pady=5)
        self.segmentation_canvas = ctk.CTkCanvas(self.segmentation_frame, width=800, height=600)
        self.segmentation_canvas.pack(pady=5)

        # Panel de seguimiento de objetos
        self.tracking_frame = ctk.CTkFrame(self.root)
        self.tracking_label = ctk.CTkLabel(self.tracking_frame, text="Seguimiento de Objetos")
        self.tracking_label.pack(pady=5)
        self.tracking_canvas = ctk.CTkCanvas(self.tracking_frame, width=800, height=600)
        self.tracking_canvas.pack(pady=5)

        # Mostrar el panel inicial (clasificación por defecto)
        self.show_classification()


    # Función para cambiar al panel clasificación
    def show_classification(self):
        self.current_process = 'Clasificación'
        self.update_ui()


    # Función para cambiar al panel detección
    def show_detection(self):
        self.current_process = 'Detección'
        self.update_ui()


    # Función para cambiar al panel segmentación
    def show_segmentation(self):
        self.current_process = 'Segmentación'
        self.update_ui()


    # Función para cambiar al panel seguimiento
    def show_tracking(self):
        self.current_process = 'Seguimiento'
        self.update_ui()


    # Función para actualizar la interfaz según el proceso seleccionado
    def update_ui(self):
        
        # Ocultar todos los paneles
        self.classification_frame.pack_forget()
        self.detection_frame.pack_forget()
        self.segmentation_frame.pack_forget()
        self.tracking_frame.pack_forget()

        # Mostrar solo el panel correspondiente al proceso actual
        if self.current_process == 'Clasificación':
            self.classification_frame.pack(fill="both", expand=True)
        elif self.current_process == 'Detección':
            self.detection_frame.pack(fill="both", expand=True)
        elif self.current_process == 'Segmentación':
            self.segmentation_frame.pack(fill="both", expand=True)
        elif self.current_process == 'Seguimiento':
            self.tracking_frame.pack(fill="both", expand=True)


    # Función para iniciar los hilos de procesamiento
    def start_processing_threads(self):
        
        # Hilo para captura continua de video
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Hilos independientes para cada tipo de procesamiento
        self.classification_thread = threading.Thread(target=self.process_classification)
        self.classification_thread.daemon = True
        self.classification_thread.start()

        # Hilos para detección, segmentación y seguimiento
        self.detection_thread = threading.Thread(target=self.process_detection)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        # Hilos para segmentación y seguimiento
        self.segmentation_thread = threading.Thread(target=self.process_segmentation)
        self.segmentation_thread.daemon = True
        self.segmentation_thread.start()

        # Hilo para seguimiento de objetos
        self.tracking_thread = threading.Thread(target=self.process_tracking)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()


    # Función para captura y procesamiento de video
    def update_video(self):
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                time.sleep(0.03)  # Limitar a ~30 FPS


    # Función para procesamiento de imágenes
    def process_classification(self):
        
        while self.running:
            if self.current_frame is not None:

                # Convertir frame a formato PIL
                image = Image.fromarray(self.current_frame)

                # Preprocesar imagen para el modelo
                input_tensor = self.transform(image)
                input_batch = input_tensor.unsqueeze(0)

                # Realizar inferencia sin calcular gradientes
                with torch.no_grad():
                    output = self.classification_model(input_batch)

                # Obtener clase con mayor probabilidad
                _, predicted = torch.max(output, 1)

                # Actualizar interfaz con resultado
                self.update_classification_ui(image, f"Clase: {predicted.item()}")

            time.sleep(0.1)  # Controlar frecuencia de procesamiento


    # Función para detección de objetos
    def process_detection(self):
        
        while self.running:
            if self.current_frame is not None:

                # Realizar detección con YOLO
                results = self.detection_model(self.current_frame)

                # Obtener frame con anotaciones de detección
                annotated_frame = results[0].plot()

                # Actualizar interfaz con resultado
                self.update_detection_ui(Image.fromarray(annotated_frame))

            time.sleep(0.1)  # Controlar frecuencia de procesamiento


    # Función para segmentación de objetos
    def process_segmentation(self):
        
        while self.running:
            if self.current_frame is not None:

                # Realizar segmentación con YOLO-seg
                results = self.segmentation_model(self.current_frame)

                # Obtener frame con máscaras de segmentación
                annotated_frame = results[0].plot()

                # Actualizar interfaz con resultado
                self.update_segmentation_ui(Image.fromarray(annotated_frame))

            time.sleep(0.1)  # Controlar frecuencia de procesamiento


    # Función para seguimiento de objetos
    def process_tracking(self):
        
        # Inicializar tracker CSRT de OpenCV
        tracker = cv2.TrackerCSRT_create()
        bbox = None

        while self.running:
            if self.current_frame is not None:

                frame = self.current_frame.copy()

                if bbox is None:
                    # Si no hay objeto siendo seguido, detectar uno nuevo
                    results = self.detection_model(frame)
                    if len(results[0].boxes) > 0:
                        # Tomar el primer objeto detectado
                        box = results[0].boxes[0].xyxy[0].numpy()
                        # Convertir formato [x1,y1,x2,y2] a [x,y,width,height]
                        bbox = (int(box[0]), int(box[1]),
                                int(box[2] - box[0]), int(box[3] - box[1]))
                        tracker.init(frame, bbox)
                else:
                    # Actualizar posición del objeto
                    success, bbox = tracker.update(frame)
                    if success:
                        # Dibujar bounding box actualizado
                        x, y, w, h = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Actualizar interfaz con resultado
                self.update_tracking_ui(Image.fromarray(frame))

            time.sleep(0.1)  # Controlar frecuencia de procesamiento


    # Función para actualizar la interfaz gráfica de clasificación
    def update_classification_ui(self, image, text):
        photo = ImageTk.PhotoImage(image.resize((800, 600)))
        self.classification_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.classification_canvas.image = photo
        self.classification_result.configure(text=text)


    # Función para actualizar la interfaz gráfica de detección
    def update_detection_ui(self, image):
        photo = ImageTk.PhotoImage(image.resize((800, 600)))
        self.detection_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.detection_canvas.image = photo


    # Función para actualizar la interfaz gráfica de segmentación
    def update_segmentation_ui(self, image):
        photo = ImageTk.PhotoImage(image.resize((800, 600)))
        self.segmentation_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.segmentation_canvas.image = photo


    # Función para actualizar la interfaz gráfica de seguimiento
    def update_tracking_ui(self, image):
        photo = ImageTk.PhotoImage(image.resize((800, 600)))
        self.tracking_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.tracking_canvas.image = photo


    # Función principal para la interfaz gráfica
    def run(self):
        self.root.mainloop()
        self.cap.release()
        cv2.destroyAllWindows()


# Punto de entrada de la aplicación
if __name__ == "__main__":
    app = VisionDashboard()
    app.run()