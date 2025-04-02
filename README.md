# Computer-Vision-Python
Este repositorio es una compilacion de varios proyectos de Computer Vision with Python and YOLO




## ComputerVision1.py
El código utiliza la librería OpenCV para detectar personas en tiempo real a través de la cámara web. Primero, se configura un descriptor HOG (Histogram of Oriented Gradients) con un detector preentrenado para la detección de personas.

Luego, se accede a la cámara web utilizando cv2.VideoCapture(0). Si la cámara no se abre correctamente, el programa se detiene. En el bucle principal, se capturan fotogramas continuamente y se aplican detecciones de personas mediante hog.detectMultiScale(), que analiza la imagen en busca de patrones característicos de figuras humanas.

Si se detectan personas, se dibujan rectángulos verdes alrededor de ellas. La imagen procesada se muestra en una ventana en tiempo real. El programa sigue ejecutándose hasta que el usuario presiona la tecla 'q', momento en el cual se libera la cámara y se cierran las ventanas de OpenCV.




## ComputerVision2.py
Este código implementa un sistema de visión computacional en tiempo real con una interfaz gráfica usando CustomTkinter. Utiliza modelos de YOLOv8 para detección de objetos, ResNet-18 para clasificación de imágenes y DeepLabV3 para segmentación, cargándolos al inicio.

La interfaz gráfica permite cambiar entre cuatro funcionalidades: clasificación de objetos, detección de personas u objetos, segmentación de imágenes y seguimiento de objetos en video. Cada opción se gestiona con un frame independiente, y se alterna entre ellos mediante botones en un menú lateral.

El código captura video en vivo desde la cámara y procesa los fotogramas en tiempo real. La clasificación utiliza ResNet-18, la detección se realiza con YOLOv8, la segmentación con DeepLabV3, y el seguimiento de objetos emplea MeanShift tras seleccionar una región de interés.

El procesamiento de imágenes usa Torchvision para preprocesamiento y normalización antes de ser enviadas a los modelos. OpenCV maneja la captura de video y el dibujado de rectángulos en detección y seguimiento. Finalmente, la aplicación inicia con la interfaz y actualiza continuamente los fotogramas hasta que el usuario la cierra.




## Bibliotecas Necesarias

- cv2

- Numpy

- Tkinter

- CustomTkinter

- Torch
 
- TorchVision

- Ultralytics

- YOLO

- Tensorflow




## Intalacion de Bibliotecas Necesarias

- pip install opencv-python
 
- pip install numpy
 
- sudo apt-get install python3-tk
 
- pip install customtkinter
 
- pip install torch
 
- pip install torchvision
 
- pip install ultralytics

- pip install YOLO

- pip install tensorflow




## Archivos de configuración de YOLO

- yolov3.weights

- yolov3.cfg
 
- coco.names




## Intalacion de Archivos de configuración de YOLO

- wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
 
- wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

- wget https://pjreddie.com/media/files/yolov3.weights
