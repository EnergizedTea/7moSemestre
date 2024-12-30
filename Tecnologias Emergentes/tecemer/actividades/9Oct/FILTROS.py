import matplotlib.pyplot as plt
import matplotlib.image as mimg
import cv2
import numpy as np

image = mimg.imread('lena.tif')

class ImageManipulation():
    def __init__(self, image):
        self.image = image

    # función que muestra y describe la imagen
    def des_img(self):
        plt.imshow(self.image)
        plt.show()

        m, n, c = self.image.shape
        print(f'Ancho:  {n}, Alto:  {m}, Canales:   {c}')
        pixel = self.image[0,0]
        print(f'Pixel (0,0): {pixel}')

    # Funcion para imprimir en terminal valores 
    def imp_val(self):
        m, n = self.image.shape
        for i in range(m):
            for j in range(n):
                pixel = self.image[i,j]
                print(pixel, end=' ')
            print() 

    # Funcion para mandar crear el histograma 
    def his_img(self):
        color = ('r','g','b')
        plt.subplot(1,2,1)
        plt.title('Histograma')
        plt.xlabel('Intensidad')
        plt.ylabel('Frecuencia')
        for i, col in enumerate(color):
            hist = cv2.calcHist([self.image], [i], None, [256], [0,256])
            plt.plot(hist, color=col)
        plt.subplot(1,2,2)
        plt.title('Imagen Original')
        plt.imshow(self.image)
        plt.show()

    # Funcion para minimizar la existencia de un color especifico
    def rgb_img(self, rgb=int):
        newimage = self.image.copy()
        m, n, c = self.image.shape
        for i in range(m):
            for j in range(n):
                newimage[i,j][rgb]= newimage[i,j][rgb]/10
        plt.imshow(newimage)
        plt.show()
        return newimage
    
    def equalize_histogram(self):
        new_image = self.image.copy()
        for i in range(3):
            new_image[:,:,i] = cv2.equalizeHist(new_image[:,:,i])
        return new_image
    
    def equalize_CLAHE(self): # CLAHE: Contrast imited Adaptive Histogram Equalization
        #Umbral para limitar el contraste
        clip = 3.0
        # Tamaño de la celda, (tilextile)
        tile = 2
        # Instanciar el objeto CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip, 
                                tileGridSize=(tile,tile))
        new_image = self.image.copy()
        # Ecualizamos la imagen
        for i in range(3):
            # : abarcar todo eso, m n y canal
            new_image[:,:,i] = clahe.apply(new_image[:,:,i])
        return new_image
    
    def cut_img(self,x1,y1,x2,y2):
        return self.image[y1:y2, x1:x2]
    
    def plot_image(self, title, second):
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
        #Mostrar imagen original
        ax1.imshow(self.image)
        ax1.set_title('Imagen Original')

        ax2.imshow(second)
        ax2.set_title(title)

        plt.tight_layout()
        plt.show()

    def rot_img(self, angle):
        m, n, c = self.image.shape
        center = (n//2, n//2)

        matriz_rotacion = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(self.image, matriz_rotacion, (n,m))
    
    def filtro(self, kernel):
        m, n, c = self.image.shape
        new_image = self.image.copy()
        for i in range(1, m -1):
            for j in range(1, n-1):
                for k in range(c):
                    suma = new_image[i-1, j-1, k] * kernel[0,0] + new_image[i-1, j, k] * kernel[0,1] + new_image[i-1, j+1, k] * kernel[0,2] + \
                    new_image[i, j-1, k] * kernel[1,0] + new_image[i, j, k] * kernel[1,1] + new_image[i, j+1, k] * kernel[1,2] + \
                    new_image[i+1, j-1, k] * kernel[2,0] + new_image[i+1, j, k] * kernel[2,1] + new_image[i+1, j+1, k] * kernel[2,2] 

                    new_image[i,j,k] = suma

        return new_image
    
    def filtro_cv2(self, kernel):
        return cv2.filter2D(self.image, -1, kernel)


imgm = ImageManipulation(image)
imgm.rgb_img(0)
newimgm = ImageManipulation(imgm.equalize_histogram())
newimgm.his_img()
new2imgm = ImageManipulation(imgm.equalize_CLAHE())
new2imgm.his_img()

cut = imgm.cut_img(169,185,366,388)
imgm.plot_image('Imagen Recortada', cut)

angled = imgm.rot_img(45)
imgm.plot_image('Imagen 45°', angled)

kernel = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]]) / 9

test = imgm.filtro_cv2(kernel)
imgm.plot_image('Filtro', test)

# TAREA DE MIERCOLES 9: TODO: Pasar a escala de grises y jugar con los jkernels para encontrar tales que 
# 1. Encuentre bordes horizontales. 
# 2. Encontrar bordes verticales.
# 3. La mezcla que detecta los bordes de todas partes


# TODO: TRABAJAR EN EL GITHUB, A LO MEJOR CREAR UNA API QUE 
# RECIBE UNA IMAGEN Y CREA UN HISTOGRAMA CON RESPECTO A ELLO
# AHORA TOCA, MANIPULACION DEL CANAL ROJO.
""" image = cv2.imread('lena.tif')
cv2.imshow('Lena', image) 
cv2.waitKey(0)
cv2.destroyAllWindows() """
# TODO: Convertir todo a funciones y crear funcion del brillo
# Deben ser modificar canal rojo, la binarización de la imagen, el canal de brillo, etc

# TODO cambiar de SELF a image = self.image PARA PERMITIR AGREGAR VALORES, 
#  esto implica poner esto como ultima condicion a agregar