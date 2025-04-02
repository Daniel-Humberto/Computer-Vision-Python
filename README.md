# Computer-Vision-Python
Este repositorio es una compilacion de varios proyectos de Computer Vision with Python and YOLO




## ComputerVision1.py
El código utiliza la librería OpenCV para detectar personas en tiempo real a través de la cámara web. Primero, se configura un descriptor HOG (Histogram of Oriented Gradients) con un detector preentrenado para la detección de personas.

Luego, se accede a la cámara web utilizando cv2.VideoCapture(0). Si la cámara no se abre correctamente, el programa se detiene. En el bucle principal, se capturan fotogramas continuamente y se aplican detecciones de personas mediante hog.detectMultiScale(), que analiza la imagen en busca de patrones característicos de figuras humanas.

Si se detectan personas, se dibujan rectángulos verdes alrededor de ellas. La imagen procesada se muestra en una ventana en tiempo real. El programa sigue ejecutándose hasta que el usuario presiona la tecla 'q', momento en el cual se libera la cámara y se cierran las ventanas de OpenCV.



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
