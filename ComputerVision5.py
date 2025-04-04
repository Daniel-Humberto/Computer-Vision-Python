from torchvision import transforms, models
from PIL import Image, ImageTk
from ultralytics import YOLO
import customtkinter as ctk
import numpy as np
import threading
import torch
import time
import cv2




class VisionDashboard:
    def __init__(self):
        # Configuración principal de la ventana
        self.root = ctk.CTk()
        self.root.title("Dashboard de Visión Computacional")
        self.root.geometry("1920x1080")

        # Cargar modelos
        self.load_models()

        # Configurar la interfaz
        self.setup_ui()

        # Inicializar la captura de video
        self.cap = cv2.VideoCapture(0)

        # Variables de control
        self.running = True
        self.current_frame = None
        self.current_mode = "classification"  # Modo inicial

        # Iniciar hilos de procesamiento
        self.start_processing_threads()

    def load_models(self):
        # Modelo de clasificación (ResNet50)
        self.classification_model = models.resnet50(pretrained=True)
        self.classification_model.eval()

        # Modelo de detección (YOLOv8)
        self.detection_model = YOLO('yolov8n.pt')

        # Modelo de segmentación (YOLOv8-seg)
        self.segmentation_model = YOLO('yolov8n-seg.pt')

        # Transformaciones para clasificación
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def setup_ui(self):
        # Configurar grid
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Frame para el menú
        menu_frame = ctk.CTkFrame(self.root)
        menu_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # Botones del menú
        modes = [
            ("Clasificación", "classification"),
            ("Detección de Objetos", "detection"),
            ("Segmentación", "segmentation"),
            ("Seguimiento de Objetos", "tracking")
        ]

        for i, (text, mode) in enumerate(modes):
            btn = ctk.CTkButton(
                menu_frame,
                text=text,
                command=lambda m=mode: self.change_mode(m)
            )
            btn.pack(side="left", padx=5, pady=5)

        # Frame principal para el video
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Canvas para el video
        self.main_canvas = ctk.CTkCanvas(self.main_frame)
        self.main_canvas.pack(expand=True, fill="both", padx=5, pady=5)

        # Label para resultados de clasificación
        self.result_label = ctk.CTkLabel(self.main_frame, text="")
        self.result_label.pack(pady=5)

    def change_mode(self, mode):
        self.current_mode = mode
        self.result_label.configure(text="")

    def start_processing_threads(self):
        # Hilo para captura de video
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Hilo para procesamiento
        self.processing_thread = threading.Thread(target=self.process_frame)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def update_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                time.sleep(0.03)  # ~30 FPS

    def process_frame(self):
        tracker = cv2.TrackerCSRT_create()
        bbox = None

        while self.running:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
                processed_frame = None

                if self.current_mode == "classification":
                    # Clasificación
                    image = Image.fromarray(frame)
                    input_tensor = self.transform(image)
                    input_batch = input_tensor.unsqueeze(0)

                    with torch.no_grad():
                        output = self.classification_model(input_batch)

                    _, predicted = torch.max(output, 1)
                    self.result_label.configure(text=f"Clase: {predicted.item()}")
                    processed_frame = frame

                elif self.current_mode == "detection":
                    # Detección
                    results = self.detection_model(frame)
                    processed_frame = results[0].plot()

                elif self.current_mode == "segmentation":
                    # Segmentación
                    results = self.segmentation_model(frame)
                    processed_frame = results[0].plot()

                elif self.current_mode == "tracking":
                    # Seguimiento
                    if bbox is None:
                        results = self.detection_model(frame)
                        if len(results[0].boxes) > 0:
                            box = results[0].boxes[0].xyxy[0].numpy()
                            bbox = (int(box[0]), int(box[1]),
                                    int(box[2] - box[0]), int(box[3] - box[1]))
                            tracker.init(frame, bbox)
                    else:
                        success, bbox = tracker.update(frame)
                        if success:
                            x, y, w, h = [int(v) for v in bbox]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    processed_frame = frame

                if processed_frame is not None:
                    self.update_main_display(Image.fromarray(processed_frame))

            time.sleep(0.1)

    def update_main_display(self, image):
        # Obtener el tamaño del canvas
        canvas_width = self.main_canvas.winfo_width()
        canvas_height = self.main_canvas.winfo_height()

        # Calcular el tamaño manteniendo la relación de aspecto
        img_width, img_height = image.size
        aspect_ratio = img_width / img_height

        if canvas_width / canvas_height > aspect_ratio:
            new_height = canvas_height
            new_width = int(canvas_height * aspect_ratio)
        else:
            new_width = canvas_width
            new_height = int(canvas_width / aspect_ratio)

        # Redimensionar la imagen
        resized_image = image.resize((new_width, new_height))
        photo = ImageTk.PhotoImage(resized_image)

        # Centrar la imagen en el canvas
        x_center = (canvas_width - new_width) // 2
        y_center = (canvas_height - new_height) // 2

        self.main_canvas.delete("all")
        self.main_canvas.create_image(x_center, y_center, anchor="nw", image=photo)
        self.main_canvas.image = photo

    def run(self):
        self.root.mainloop()

    def cleanup(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = VisionDashboard()
    try:
        app.run()
    finally:
        app.cleanup()