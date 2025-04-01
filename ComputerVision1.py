# Importamos la librería OpenCV para procesamiento de imágenes y video
import cv2




# Creamos un descriptor de Histogram of Oriented Gradients (HOG) para la detección de personas
hog = cv2.HOGDescriptor()

# Configuramos el descriptor HOG con el detector de personas preentrenado
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Capturamos el video desde la cámara web (ID 0)
cap = cv2.VideoCapture(0)


# Verificamos si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara web")
    exit()

# Bucle principal para capturar y procesar los fotogramas en tiempo real
while True:
    ret, frame = cap.read()  # Leemos un fotograma de la cámara

    # Si no se pudo leer un fotograma, salimos del bucle
    if not ret:
        break

    # Aplicamos el detector de personas en el fotograma
    (rects, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # Dibujamos rectángulos alrededor de las personas detectadas
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Color verde, grosor de línea 2

    # Mostramos el fotograma con las detecciones
    cv2.imshow('Deteccion de Personas en Tiempo Real', frame)

    # Si el usuario presiona la tecla 'q', terminamos el bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos la cámara y cerramos las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()