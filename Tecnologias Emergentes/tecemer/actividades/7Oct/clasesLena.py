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
    
    # Funcion para binarizar imagen
    def bin_img(self):
        gray_image = np.mean(self.image, axis=2)
        bin_image = (gray_image > 128) * 255
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


imgm = ImageManipulation(image)
imgm.bri_img(20)

""" image = cv2.imread('lena.tif')
cv2.imshow('Lena', image) 
cv2.waitKey(0)
cv2.destroyAllWindows() """

# TODO: Convertir todo a funciones y crear funcion del brillo
# Deben ser modificar canal rojo, la binarización de la imagen, el canal de brillo, etc