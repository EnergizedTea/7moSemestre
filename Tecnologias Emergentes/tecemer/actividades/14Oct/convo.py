import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import cv2

image = mimg.imread('lena.tif')

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
        plt.title(f'Size {pool_m + 1} x {pool_n + 1}')
        plt.show()

        return max_pool 
    
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
                # print(result[i,j])
        for i in range(6):
            result = self.max_pool(result)
        return result
        

imgm = ImageManipulation(image)
imgm.set_img(imgm.bnw_img())
imgm.con_img()

""" image = cv2.imread('lena.tif')
cv2.imshow('Lena', image) 
cv2.waitKey(0)
cv2.destroyAllWindows() """
