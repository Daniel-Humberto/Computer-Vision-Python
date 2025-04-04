import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import PIL.Image, PIL.ImageTk
import customtkinter as ctk
import numpy as np
import threading
import time
import cv2




class VisionDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Vision Computacional en Vivo")
        self.geometry("1280x720")

        # Inicialización de variables para el video
        self.video_source = 0  # Default camera
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {self.video_source}")
            exit()

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Modelo de Clasificación (ejemplo MobileNetV2)
        self.classification_model = load_model("modelo_clasificacion.h5")  # Reemplaza con tu modelo
        self.classification_labels = ["Clase1", "Clase2", "Clase3"]  # Reemplaza con tus etiquetas

        # Modelo de Detección (ejemplo SSD MobileNet)
        self.detection_model = load_model("modelo_deteccion.h5") # Reemplaza con tu modelo
        self.detection_labels = ["Objeto1", "Objeto2", "Objeto3"]  # Reemplaza con tus etiquetas
        self.confidence_threshold = 0.5

        # Modelo de Segmentación (ejemplo DeepLabV3+)
        self.segmentation_model = load_model("modelo_segmentacion.h5") # Reemplaza con tu modelo
        self.segmentation_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Reemplaza con tus colores

        # Variables para seguimiento de objetos (se inicializan en el frame correspondiente)
        self.tracker = None  # Inicializar rastreador

        # Configuración de la interfaz
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        self.create_frames()

        # Inicia la actualización del video
        self.update_all_frames()  # Iniciar la captura y procesamiento de frames

    def create_frames(self):
        # Frame de Clasificación
        self.classification_frame = ctk.CTkFrame(self)
        self.classification_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.classification_label = ctk.CTkLabel(self.classification_frame, text="Clasificación")
        self.classification_label.pack(pady=5)
        self.classification_canvas = ctk.CTkCanvas(self.classification_frame, width=self.width, height=self.height)
        self.classification_canvas.pack()
        self.classification_result_label = ctk.CTkLabel(self.classification_frame, text="Resultado: ")
        self.classification_result_label.pack(pady=5)

        # Frame de Detección
        self.detection_frame = ctk.CTkFrame(self)
        self.detection_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.detection_label = ctk.CTkLabel(self.detection_frame, text="Detección de Objetos")
        self.detection_label.pack(pady=5)
        self.detection_canvas = ctk.CTkCanvas(self.detection_frame, width=self.width, height=self.height)
        self.detection_canvas.pack()
        self.detection_result_label = ctk.CTkLabel(self.detection_frame, text="Objetos detectados: ")
        self.detection_result_label.pack(pady=5)

        # Frame de Segmentación
        self.segmentation_frame = ctk.CTkFrame(self)
        self.segmentation_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.segmentation_label = ctk.CTkLabel(self.segmentation_frame, text="Segmentación")
        self.segmentation_label.pack(pady=5)
        self.segmentation_canvas = ctk.CTkCanvas(self.segmentation_frame, width=self.width, height=self.height)
        self.segmentation_canvas.pack()

        # Frame de Seguimiento
        self.tracking_frame = ctk.CTkFrame(self)
        self.tracking_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.tracking_label = ctk.CTkLabel(self.tracking_frame, text="Seguimiento de Objetos")
        self.tracking_label.pack(pady=5)
        self.tracking_canvas = ctk.CTkCanvas(self.tracking_frame, width=self.width, height=self.height)
        self.tracking_canvas.pack()
        self.tracking_button = ctk.CTkButton(self.tracking_frame, text="Seleccionar Objeto", command=self.select_object_to_track)
        self.tracking_button.pack(pady=5)
        self.tracking_object_label = ctk.CTkLabel(self.tracking_frame, text="Objeto rastreado: Ninguno")
        self.tracking_object_label.pack(pady=5)

        self.tracking_canvas.bind("<Button-1>", self.start_tracking) # Binding del evento click

    def update_all_frames(self):
        try:
            ret, frame = self.cap.read()  # Capture frame-by-frame

            if ret:
                # Clasificación
                self.process_classification(frame)
                # Detección
                self.process_detection(frame)
                # Segmentación
                self.process_segmentation(frame)
                # Seguimiento
                self.process_tracking(frame)

            self.after(10, self.update_all_frames)  # Update every 10 milliseconds

        except Exception as e:
            print(f"Error en la función 'update_all_frames': {e}")


    def process_classification(self, frame):
        try:
            # Preprocesamiento de la imagen para clasificación
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))  # Ajusta al tamaño esperado por el modelo
            img = img_to_array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            # Predicción
            predictions = self.classification_model.predict(img)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]

            # Mostrar resultado
            label = f"{self.classification_labels[predicted_class]} ({confidence:.2f})"
            self.classification_result_label.configure(text=f"Clasificación: {label}")

            # Mostrar la imagen
            self.display_image(frame, self.classification_canvas)
        except Exception as e:
            print(f"Error en la función 'process_classification': {e}")

    def process_detection(self, frame):
        try:
            # Preprocesamiento de la imagen para detección
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape
            input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)

            # Predicción
            detections = self.detection_model.signatures['serving_default'](input_tensor)

            # Procesamiento de resultados
            boxes = detections['detection_boxes'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(np.int32)
            scores = detections['detection_scores'][0].numpy()

            # Dibujar bounding boxes
            for i in range(len(scores)):
                if scores[i] > self.confidence_threshold:
                    ymin, xmin, ymax, xmax = boxes[i]
                    xmin = int(xmin * image_width)
                    xmax = int(xmax * image_width)
                    ymin = int(ymin * image_height)
                    ymax = int(ymax * image_height)

                    class_id = classes[i]
                    label = self.detection_labels[class_id - 1]  # Ajustar índice
                    confidence = scores[i]

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Mostrar la imagen con las detecciones
            self.display_image(frame, self.detection_canvas)
            self.detection_result_label.configure(text=f"Objetos detectados: {len([s for s in scores if s > self.confidence_threshold])}")
        except Exception as e:
            print(f"Error en la función 'process_detection': {e}")

    def process_segmentation(self, frame):
        try:
            # Preprocesamiento de la imagen para segmentación
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))  # Ajusta al tamaño esperado por el modelo
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32) / 255.0

            # Predicción
            segmentation_mask = self.segmentation_model.predict(img)[0]
            segmentation_mask = np.argmax(segmentation_mask, axis=-1)

            # Crear una imagen de color a partir de la máscara de segmentación
            colored_mask = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
            for class_id, color in enumerate(self.segmentation_colors):
                colored_mask[segmentation_mask == class_id] = color

            # Redimensionar la máscara coloreada al tamaño original del frame
            colored_mask = cv2.resize(colored_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Superponer la máscara coloreada sobre el frame original
            frame = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)

            # Mostrar la imagen segmentada
            self.display_image(frame, self.segmentation_canvas)

        except Exception as e:
            print(f"Error en la función 'process_segmentation': {e}")

    def process_tracking(self, frame):
        try:
            if self.tracker:
                success, box = self.tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    self.tracking_object_label.configure(text="Objeto perdido")

            self.display_image(frame, self.tracking_canvas)

        except Exception as e:
            print(f"Error en la función 'process_tracking': {e}")


    def display_image(self, frame, canvas):
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame)
            img = PIL.ImageTk.PhotoImage(image=img)
            canvas.img = img  # Keep a reference!
            canvas.create_image(0, 0, image=img, anchor=ctk.NW)
        except Exception as e:
            print(f"Error en la función 'display_image': {e}")


    def select_object_to_track(self):
        self.tracking_object_label.configure(text="Selecciona un objeto en el frame")
        self.tracking_canvas.bind("<Button-1>", self.start_tracking)

    def start_tracking(self, event):
        x = event.x
        y = event.y
        width = 50  # Tamaño del bounding box inicial
        height = 50
        bbox = (x - width / 2, y - height / 2, width, height)

        ret, frame = self.cap.read()
        if not ret:
            return

        self.tracker = cv2.TrackerKCF_create()  # Inicializa el rastreador KCF
        success = self.tracker.init(frame, bbox)

        if success:
             self.tracking_object_label.configure(text="Rastreando objeto")

        self.tracking_canvas.unbind("<Button-1>") # Remover el bind temporal



    def on_closing(self):
        print("Cerrando...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.destroy()


if __name__ == "__main__":
    app = VisionDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)  # Manejo del cierre de la ventana
    app.mainloop()