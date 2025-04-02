# Computer-Vision-Python
Este repositorio es una compilacion de varios proyectos de Computer Vision with Python and YOLO




## ComputerVision1.py
El código utiliza la librería OpenCV para detectar personas en tiempo real a través de la cámara web. Primero, se configura un descriptor HOG (Histogram of Oriented Gradients) con un detector preentrenado para la detección de personas.

Luego, se accede a la cámara web utilizando cv2.VideoCapture(0). Si la cámara no se abre correctamente, el programa se detiene. En el bucle principal, se capturan fotogramas continuamente y se aplican detecciones de personas mediante hog.detectMultiScale(), que analiza la imagen en busca de patrones característicos de figuras humanas.

Si se detectan personas, se dibujan rectángulos verdes alrededor de ellas. La imagen procesada se muestra en una ventana en tiempo real. El programa sigue ejecutándose hasta que el usuario presiona la tecla 'q', momento en el cual se libera la cámara y se cierran las ventanas de OpenCV.




## ComputerVision3.py
Este código implementa un sistema de visión computacional en tiempo real con una interfaz gráfica usando CustomTkinter. Utiliza modelos de YOLOv8 para detección de objetos, ResNet-18 para clasificación de imágenes y DeepLabV3 para segmentación, cargándolos al inicio.

La interfaz gráfica permite cambiar entre cuatro funcionalidades: clasificación de objetos, detección de personas u objetos, segmentación de imágenes y seguimiento de objetos en video. Cada opción se gestiona con un frame independiente, y se alterna entre ellos mediante botones en un menú lateral.

El código captura video en vivo desde la cámara y procesa los fotogramas en tiempo real. La clasificación utiliza ResNet-18, la detección se realiza con YOLOv8, la segmentación con DeepLabV3, y el seguimiento de objetos emplea MeanShift tras seleccionar una región de interés.

El procesamiento de imágenes usa Torchvision para preprocesamiento y normalización antes de ser enviadas a los modelos. OpenCV maneja la captura de video y el dibujado de rectángulos en detección y seguimiento. Finalmente, la aplicación inicia con la interfaz y actualiza continuamente los fotogramas hasta que el usuario la cierra.




## ComputerVision4.py
Este código implementa un dashboard de visión computacional en Python utilizando modelos de inteligencia artificial para clasificación, detección, segmentación y seguimiento de objetos en tiempo real. La interfaz gráfica está construida con CustomTkinter (ctk), mientras que los modelos de IA provienen de Torchvision y YOLOv8.

Al iniciar, el programa carga modelos preentrenados, incluyendo ResNet50 para clasificación de imágenes y YOLOv8 para detección y segmentación. También configura la interfaz gráfica, dividiéndola en cuatro secciones: clasificación, detección, segmentación y seguimiento.

El programa captura video en tiempo real desde la cámara usando OpenCV y lo procesa en hilos paralelos para evitar bloqueos. En cada hilo, se realizan inferencias con los modelos correspondientes:
Finalmente, el programa actualiza la interfaz gráfica en tiempo real con los resultados procesados. Al cerrarse, libera los recursos como la cámara y las ventanas de OpenCV.




## ComputerVision4.2.py
Este código implementa un dashboard de visión computacional en Python utilizando CustomTkinter para la interfaz gráfica y modelos de inteligencia artificial para realizar tareas de clasificación, detección, segmentación y seguimiento de objetos.

El constructor de la clase VisionDashboard configura la aplicación, inicializa la cámara web y carga los modelos de IA: ResNet50 para clasificación y YOLOv8 para detección y segmentación. Además, inicia hilos de procesamiento para cada tarea, asegurando la ejecución en paralelo.

La interfaz gráfica permite cambiar entre las diferentes funcionalidades mediante botones de navegación. Cada funcionalidad tiene su propio panel, que se actualiza dinámicamente según la selección del usuario.

El código captura imágenes en tiempo real desde la webcam y las procesa con los modelos. Para la clasificación, convierte los frames en tensores y los pasa por ResNet50. En la detección y segmentación, usa YOLO para identificar objetos y superponer anotaciones en la imagen. Para el seguimiento, emplea OpenCV Tracker CSRT, detectando un objeto y manteniendo su posición en la imagen.

Finalmente, los resultados se muestran en la interfaz en tiempo real mediante PhotoImage de PIL, actualizando los paneles correspondientes con las imágenes procesadas.




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
