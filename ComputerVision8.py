import torch
from torchvision import models, transforms
import tensorflow as tf
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np
import cv2





ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# =============================================
# Modelos preentrenados (sin archivos externos)
# =============================================

# 1. Clasificación (MobileNetV2)
classifier = tf.keras.applications.MobileNetV2(weights="imagenet")

# 2. Detección de Objetos (SSD MobileNet)
ssd_net = cv2.dnn.readNetFromTensorflow(
    "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
    "models/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
)

# 3. Segmentación (DeepLabV3)
deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True)
deeplab.eval()

# 4. Seguimiento (Tracker KCF)
tracker = cv2.TrackerKCF_create()
tracking = False


# =============================================
# Funciones de procesamiento
# =============================================

def classify_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_array = tf.keras.applications.mobilenet_v2.preprocess_input(frame_resized)
    predictions = classifier.predict(np.expand_dims(frame_array, axis=0))
    return tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0][1]


def detect_objects(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True)
    ssd_net.setInput(blob)
    detections = ssd_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


def segment_frame(frame):
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)

    with torch.no_grad():
        output = deeplab(input_tensor)['out'][0]
    output_mask = output.argmax(0).cpu().numpy()

    mask_colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    return mask_colors[output_mask]


def start_tracking(frame):
    global tracker, tracking
    bbox = cv2.selectROI("Seleccionar Objeto", frame, False)
    tracker.init(frame, bbox)
    tracking = True
    cv2.destroyWindow("Seleccionar Objeto")


def track_object(frame):
    if tracking:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return frame


# =============================================
# Interfaz Gráfica
# =============================================
class VisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Visión Computacional Integrada")
        self.geometry("1280x720")

        # Configurar grid
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        # Crear widgets
        self.create_frames()
        self.create_buttons()

        # Iniciar video
        self.cap = cv2.VideoCapture(0)
        self.update_frames()

    def create_frames(self):
        self.frame_classify = ctk.CTkFrame(self, width=400, height=300)
        self.frame_detect = ctk.CTkFrame(self, width=400, height=300)
        self.frame_segment = ctk.CTkFrame(self, width=400, height=300)
        self.frame_track = ctk.CTkFrame(self, width=400, height=300)

        self.frame_classify.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.frame_detect.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.frame_segment.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.frame_track.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.labels = {}
        for frame, name in zip(
                [self.frame_classify, self.frame_detect, self.frame_segment, self.frame_track],
                ["Clasificación", "Detección", "Segmentación", "Seguimiento"]
        ):
            label = ctk.CTkLabel(frame, text="")
            label.pack(expand=True, fill="both")
            self.labels[name] = label

    def create_buttons(self):
        self.btn_track = ctk.CTkButton(
            self.frame_track,
            text="Iniciar Seguimiento",
            command=lambda: start_tracking(self.current_frame)
        )
        self.btn_track.pack(pady=5)

    def update_frames(self):
        ret, frame = self.cap.read()
        if not ret: return

        self.current_frame = frame.copy()
        processed = {
            "Clasificación": self.process_classify(frame),
            "Detección": self.process_detect(frame),
            "Segmentación": self.process_segment(frame),
            "Seguimiento": self.process_track(frame)
        }

        for name, img in processed.items():
            self.display_image(img, self.labels[name])

        self.after(10, self.update_frames)

    def process_classify(self, frame):
        label = classify_frame(frame)
        frame_out = frame.copy()
        cv2.putText(frame_out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame_out

    def process_detect(self, frame):
        return detect_objects(frame.copy())

    def process_segment(self, frame):
        return segment_frame(frame.copy())

    def process_track(self, frame):
        return track_object(frame.copy())

    def display_image(self, frame, label_widget):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label_widget.configure(image=imgtk)
        label_widget.image = imgtk

    def on_closing(self):
        self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = VisionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()