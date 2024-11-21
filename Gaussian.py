import cv2 as cv
import numpy as np
from scipy.signal import convolve2d
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def callback(input):
    pass

def gaussianKernel(size, sigma):
    """
    Genera un kernel gaussiano bidimensional de tamaño size y desviación estándar sigma.
    """
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))  # Corregido
    return kernel / kernel.sum()  # Normalizamos el kernel

def applyGaussianFilterVectorized(image, kernel):
    """
    Aplica un filtro gaussiano utilizando operaciones vectorizadas con scipy.signal.convolve2d.
    """
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Aplicar convolución 2D
    filtered = convolve2d(image, kernel, mode='same', boundary='symm')
    return np.clip(filtered, 0, 255).astype(np.uint8)

def gaussianFiltering():
    """
    Aplicación del filtro gaussiano interactivo.
    """
    # File chooser para seleccionar una imagen
    Tk().withdraw()  # Oculta la ventana principal de tkinter
    imgPath = askopenfilename(title="Seleccione una imagen", filetypes=[("Archivos de imagen", ".bmp;.jpg;*.png")])
    
    if not imgPath:
        print("No se seleccionó ningún archivo. Saliendo...")
        return

    # Cargar la imagen seleccionada
    img = cv.imread(imgPath)

    # Parámetros iniciales
    kernel_size = 21  # Tamaño del kernel
    winName = 'Gaussian Filter'
    cv.namedWindow(winName)
    cv.createTrackbar('sigma', winName, 1, 20, callback)
    
    # Escalar la imagen para mejorar rendimiento
    scale = 0.5
    img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        # Leer el valor actual de sigma desde el deslizador
        sigma = cv.getTrackbarPos('sigma', winName)
        if sigma == 0:
            sigma = 1

        # Crear el kernel gaussiano
        kernel = gaussianKernel(kernel_size, sigma)

        # Aplicar el filtro gaussiano
        imgFilter = applyGaussianFilterVectorized(img, kernel)

        # Mostrar el resultado
        cv.imshow(winName, imgFilter)

    cv.destroyAllWindows()

if __name__ == '__main__': 
    gaussianFiltering()
