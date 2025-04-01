import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np
import torch
import cv2




# Carga los modelos de detección, clasificación y segmentación
yolo_model = YOLO("yolov8n.pt")
classification_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
classification_model.eval()
segmentation_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
segmentation_model.eval()


# Función para preprocesar la imagen antes de la clasificación o segmentación
def preprocess_image(image):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(image).unsqueeze(0)


# Configuración de la interfaz gráfica
ctk.set_appearance_mode("dark")
app = ctk.CTk()
app.geometry("1000x600")
app.title("Visión Computacional en Vivo")


# Creación de los frames para cada funcionalidad
frame_classification = ctk.CTkFrame(app)
frame_detection = ctk.CTkFrame(app)
frame_segmentation = ctk.CTkFrame(app)
frame_tracking = ctk.CTkFrame(app)


# Función para mostrar solo el frame seleccionado
def show_frame(frame):

    for f in [frame_classification, frame_detection, frame_segmentation, frame_tracking]:
        f.pack_forget()
    frame.pack(fill="both", expand=True)

# Menú lateral con botones para cambiar de funcionalidad
menu = ctk.CTkFrame(app, width=200)
menu.pack(side="left", fill="y")
btn_classify = ctk.CTkButton(menu, text="Clasificación", command=lambda: show_frame(frame_classification))
btn_classify.pack(pady=10)
btn_detect = ctk.CTkButton(menu, text="Detección", command=lambda: show_frame(frame_detection))
btn_detect.pack(pady=10)
btn_segment = ctk.CTkButton(menu, text="Segmentación", command=lambda: show_frame(frame_segmentation))
btn_segment.pack(pady=10)
btn_track = ctk.CTkButton(menu, text="Seguimiento", command=lambda: show_frame(frame_tracking))
btn_track.pack(pady=10)

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)


# Función para realizar clasificación de objetos en video
def classify_video():

    ret, frame = cap.read()

    if ret:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess_image(img)
        output = classification_model(input_tensor)
        class_index = output.argmax().item()
        result_text.configure(text=f"Clase detectada: {class_index}")
    frame_classification.after(30, classify_video)

result_text = ctk.CTkLabel(frame_classification, text="Esperando clasificación...")
result_text.pack()


# Función para detectar objetos en video usando YOLO
def detect_objects():

    ret, frame = cap.read()

    if ret:
        results = yolo_model(frame)
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        lbl_detected.configure(image=img)
        lbl_detected.image = img
    frame_detection.after(30, detect_objects)

lbl_detected = ctk.CTkLabel(frame_detection)
lbl_detected.pack()


# Función para segmentar una imagen en tiempo real
def segment_image():

    ret, frame = cap.read()

    if ret:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess_image(img)
        output = segmentation_model(input_tensor)['out']
        mask = output.argmax(1).squeeze().detach().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        img = ImageTk.PhotoImage(Image.fromarray(mask))
        lbl_segmented.configure(image=img)
        lbl_segmented.image = img
    frame_segmentation.after(100, segment_image)

# Etiqueta para mostrar la imagen segmentada
lbl_segmented = ctk.CTkLabel(frame_segmentation)
lbl_segmented.pack()

# Variables para el seguimiento de objetos
tracking_object = False
track_window = None
roi_hist = None


# Función para realizar seguimiento de objetos en video
def track_objects():

    global tracking_object, track_window, roi_hist

    ret, frame = cap.read()

    if not ret:
        frame_tracking.after(30, track_objects)
        return

    if tracking_object and track_window is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.meanShift(dst, track_window, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    lbl_tracked.configure(image=img)
    lbl_tracked.image = img
    frame_tracking.after(30, track_objects)


# Función para seleccionar un objeto a rastrear
def select_object():

    global tracking_object, track_window, roi_hist

    ret, frame = cap.read()

    if not ret:
        return
    r = cv2.selectROI("Selecciona objeto", frame, fromCenter=False)
    if r[2] > 0 and r[3] > 0:
        track_window = r
        roi = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        tracking_object = True
    cv2.destroyWindow("Selecciona objeto")

btn_select = ctk.CTkButton(frame_tracking, text="Seleccionar Objeto", command=select_object)
btn_select.pack()
lbl_tracked = ctk.CTkLabel(frame_tracking)
lbl_tracked.pack()


# Inicializar la interfaz con la primera opción
show_frame(frame_classification)


# Ejecutar las funciones de procesamiento en tiempo real
detect_objects()
classify_video()
segment_image()
track_objects()


# Iniciar la aplicación
detect_objects()
app.mainloop()
cap.release()
cv2.destroyAllWindows()