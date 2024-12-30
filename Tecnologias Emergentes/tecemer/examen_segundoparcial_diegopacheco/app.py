import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import cv2

class ImageManipulation():
    def __init__(self, image):
        self.image = image

    # Funcion que cambia la imagen con la que se trabaja
    def set_img(self, image):
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

    # Funcion para convertir imagen a blanco y negro
    def bnw_img(self):
        bnw_image = np.mean(self.image, axis=2)
        plt.imshow(bnw_image, cmap='gray')
        plt.show()
        return bnw_image
    
    # Funcion para binarizar imagen
    def bin_img(self):
        bnw_image = self.bnw_img()
        bin_image = (bnw_image > 128) * 255
        plt.imshow(bin_image, cmap='gray')
        plt.show()
        return bin_image

    # Funcion para aumentar el canal de brillo
    def bri_img(self, valor):
        bri_image = self.image.copy()
        bri_image = bri_image + valor
        plt.imshow(bri_image)
        plt.show()
        return bri_image
    
    # Funcion para aplicar convolucion (checar notas del 14 de Octubre)
    def con_img(self, kernel=np.ones((3,3))):
        image = self.image.copy()
        m, n = image.shape
        result = np.zeros((m-1,n-1))
        for i in range(m-1):
            for j in range(n-1):
                section = image[i:i+3, j:j+3]
                # print(f'start, i = {i} and j = {j}')
                value = 0
                for a, fila in enumerate(section):
                    for b, valor in enumerate(fila):
                        value += valor * kernel[a][b]
                result[i, j] = round(value/9)
        return result
    
    def max_pool(self, image = None, pool_size=2):
        if image is None:
            image = self.image.copy()

        m, n = image.shape
        pool_m = m // pool_size
        pool_n = n // pool_size

        max_pool = np.zeros((pool_m, pool_n))

        for i in range(0, m - pool_size + 1, pool_size):
            for j in range(0, n - pool_size + 1, pool_size):
                pool = image[i:i+pool_size, j:j+pool_size]
                max_pool[i//pool_size, j//pool_size] = np.max(pool)

        # Show the result
        plt.imshow(max_pool, cmap='gray')
        # plt.title(f'Size {pool_m + 1} x {pool_n + 1}')
        # plt.show()

        return max_pool 
    
    def con_img2(self, kernel=np.ones((3,3))):
        image = self.image.copy()
        m, n = image.shape
        result = np.zeros((m-1,n-1))
        for i in range(m-1):
            for j in range(n-1):
                section = image[i:i+3, j:j+3]
                # print(f'start, i = {i} and j = {j}')
                value = 0
                for a, fila in enumerate(section):
                    for b, valor in enumerate(fila): 
                        value += valor * kernel[a][b]
                result[i, j] = round(value/9)
                # print(result[i,j])
        for i in range(2):
            result = self.max_pool(result)
        return result

    # Funcion para identificar bordes de imagen.
    def bor_img(self):
        # self.image = self.bnw_img()
        hor_ker = [[-1, -1, -1], 
                   [2, 2, 2],
                   [-1, -1, -1]]
        
        ver_ker = [[-1, 2, -1],
                   [-1, 2, -1],
                   [-1, 2, -1]]
        
        two_ker = [[-1, -1, -1],
                   [-1,  16, -1],
                   [-1, -1, -1]]
        
        hor_image = self.con_img(hor_ker)
        ver_image = self.con_img(ver_ker)
        two_image = self.con_img(two_ker)
        plt.figure(figsize=(12, 6))
    
        plt.subplot(2, 2, 1)  
        plt.imshow(self.image, cmap='gray')
        plt.title('Imagen Original')
        
        plt.subplot(2, 2, 2)
        plt.imshow(hor_image, cmap='gray')
        plt.title('Bordes Horizontales')
        
        plt.subplot(2, 2, 3)
        plt.imshow(ver_image, cmap='gray')
        plt.title('Bordes Verticales')
        
        plt.subplot(2, 2, 4)
        plt.imshow(two_image, cmap='gray')
        plt.title('Bordes Bordes')
        
        plt.tight_layout()

        plt.show()
        return two_image

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def rot_img(self, angle):
        m, n, c = self.image.shape
        center = (n//2, n//2)

        matriz_rotacion = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(self.image, matriz_rotacion, (n,m))

# 1 Preprocesamiento de la imagen de la placa
# A)
image = mimg.imread('test-car.jpg')
# B)
img = ImageManipulation(image)
# img.plot_image()
img.set_img(img.equalize_CLAHE())
# img.plot_image()
img.set_img(img.bin_img())
# C)

# 2 Segmentación de la placa
# A mas o menos
plaque = ImageManipulation(img.cut_img(474, 638, 788, 702))
# plaque.plot_image()
plaque.set_img(plaque.con_img2())
plaque.plot_image()
# B
char1 = plaque.cut_img(5, 0, 14, 14)
char2 = plaque.cut_img(16, 1, 24, 14)
char3 = plaque.cut_img(27, 0, 36, 14)
char4 = plaque.cut_img(41, 1, 49, 14)
char5 = plaque.cut_img(52, 1, 61, 14)
char6 = plaque.cut_img(62, 1, 71, 14)

'''plt.imshow(char1)
plt.show()
plt.imshow(char2)
plt.show()
plt.imshow(char3)
plt.show()
plt.imshow(char4)
plt.show()
plt.imshow(char5)
plt.show()
plt.imshow(char6)
plt.show()'''


